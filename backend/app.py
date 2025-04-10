import os
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import httpx
import time
from dotenv import load_dotenv
from collections import deque

import brain

load_dotenv()

app = FastAPI(title="SHL Solutions RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# map test_type initials to full names
def map_test_type_to_full_name(test_type: str) -> str:
    """Maps test type initials to their full names."""
    type_map = {
        "A": "Ability & Aptitude",
        "B": "Biodata & Situational Judgement",
        "C": "Competencies",
        "D": "Development & 360",
        "E": "Assessment Exercises",
        "K": "Knowledge & Skills",
        "P": "Personality & Behavior",
        "S": "Simulations",
    }

    if not test_type:
        return "Not specified"

    result = []
    for char in test_type:
        if char in type_map:
            result.append(type_map[char])

    if result:
        return ", ".join(result)

    return test_type


TPM = 4000  # Tokens per minute
TOKENS_WINDOW_SIZE = 60  # Window size in seconds
token_bucket = deque(maxlen=1000)


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    stream: Optional[bool] = False


class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage


class RecommendationItem(BaseModel):
    name: str
    url: str
    remote_testing: str
    adaptive_irt: str
    test_type: str


class RecommendationsResponse(BaseModel):
    recommendations: List[RecommendationItem]


class AssessmentItem(BaseModel):
    url: str
    adaptive_support: str
    description: str
    duration: int = 60
    remote_support: str
    test_type: List[str]


class QueryResponse(BaseModel):
    recommended_assessments: List[AssessmentItem]


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


@app.get("/health")
async def health_check():
    return {"status": "healthy", "config": {"tpm": TPM}}


async def get_llm_response(
    query: str, context: str, model: str = "llama-3.3-70b-versatile"
) -> str:
    """Get response from Groq LLM"""
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set")

    system_prompt = """You are an AI assistant that helps recommend SHL test solutions based on user queries.
    Given the context information about various SHL test solutions, provide recommendations in a clear format.
    Focus only on the most relevant solutions that match the user's needs."""

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Here is context information about SHL solutions:\n\n{context}\n\nBased on this information, answer the following query:\n{query}",
        },
    ]

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                GROQ_API_URL,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": 0.2,
                    "max_tokens": 1000,
                },
                timeout=30.0,
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Error from Groq API: {response.text}",
                )

            result = response.json()
            return result["choices"][0]["message"]["content"]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling Groq API: {str(e)}")


async def check_rate_limit(request: Request):
    """Check if the current request exceeds the rate limit"""
    now = time.time()

    one_minute_ago = now - TOKENS_WINDOW_SIZE
    while token_bucket and token_bucket[0] < one_minute_ago:
        token_bucket.popleft()

    current_usage = len(token_bucket)

    if current_usage < TPM:
        return True

    raise HTTPException(
        status_code=429,
        detail=f"Rate limit exceeded. TPM limit: {TPM}",
        headers={"Retry-After": "60"},
    )


@app.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
    dependencies=[Depends(check_rate_limit)],
)
async def chat_completion(request: ChatCompletionRequest):
    """OpenAI-compatible endpoint for chat completions using RAG with Groq's LLM"""
    try:
        query = ""
        for message in reversed(request.messages):
            if message.role == "user":
                query = message.content
                break

        if not query:
            raise HTTPException(status_code=400, detail="No user message found")

        search_results = brain.search_astra(query, limit=10)

        if (
            not search_results
            or not hasattr(search_results, "source_nodes")
            or not search_results.source_nodes
        ):
            context = "No relevant solutions found."
            recommendations = []
        else:
            context_parts = []
            recommendations = []

            for node in search_results.source_nodes:
                meta = node.metadata

                if meta.get("name"):
                    test_type = meta.get("test_type", "Not specified")
                    full_test_type = map_test_type_to_full_name(test_type)

                    recommendations.append(
                        {
                            "name": meta.get("name", "Unknown"),
                            "url": meta.get("link", "Not available"),
                            "remote_testing": meta.get("remote_testing", "No"),
                            "adaptive_irt": meta.get("adaptive_irt", "No"),
                            "test_type": full_test_type,
                        }
                    )

                context_parts.append(node.text)

            unique_recommendations = {}
            for rec in recommendations:
                if rec["name"] not in unique_recommendations:
                    unique_recommendations[rec["name"]] = rec

            recommendations = list(unique_recommendations.values())

            recommendations = recommendations[:10]

            context = "\n---\n".join(context_parts)

        # Check if the query is a general greeting or conversation starter
        greeting_patterns = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening", "how are you"]
        is_greeting = query.lower().strip() in greeting_patterns or query.lower().strip().startswith(tuple(greeting_patterns))

        if is_greeting:
            # For greetings, return a conversational response instead of recommendations
            prompt = f"""The user said: "{query}". This appears to be a greeting or general conversation starter.
            
            Please respond with a friendly greeting and briefly explain what this API can help with:
            
            "This API can help you find SHL assessment solutions based on your hiring or development needs. 
            Could you provide details about what type of assessment you're looking for? 
            For example, you might ask about assessments for software developers, customer service roles, 
            or specific skills and competencies."
            """
            
            est_input_tokens = len(query) // 4  # Simple estimate
            llm_response = await get_llm_response(prompt, "", model=request.model)
            est_output_tokens = len(llm_response) // 4  # Simple estimate
            
            # Rest of the function continues as normal
            total_tokens = est_input_tokens + est_output_tokens

            now = time.time()
            for _ in range(total_tokens):
                token_bucket.append(now)

            return ChatCompletionResponse(
                id=f"chatcmpl-{os.urandom(4).hex()}",
                object="chat.completion",
                created=int(now),
                model=request.model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=Message(role="assistant", content=llm_response),
                        finish_reason="stop",
                    )
                ],
                usage=ChatCompletionUsage(
                    prompt_tokens=est_input_tokens,
                    completion_tokens=est_output_tokens,
                    total_tokens=total_tokens,
                ),
            )

        # For regular queries, continue with the normal recommendation flow
        recommendations_table = "Here are the recommended SHL test solutions:\n\n"
        recommendations_table += (
            "| Assessment Name | URL | Remote Testing | Adaptive/IRT | Test Type |\n"
        )
        recommendations_table += "| --- | --- | --- | --- | --- |\n"

        for rec in recommendations:
            recommendations_table += f"| {rec['name']} | {rec['url']} | {rec['remote_testing']} | {rec['adaptive_irt']} | {rec['test_type']} |\n"

        if not recommendations:
            recommendations_table = "No recommendations found matching your query."

        prompt = f"""Based on the query: "{query}", provide a helpful response recommending the most relevant SHL test solutions.
        Present the recommendations in a clear tabular format with these details:
        - Assessment name with URL
        - Remote Testing Support
        - Adaptive/IRT Support
        - Test Type
        
        Here's the data to include in your response:
        {recommendations_table}
        
        Provide a brief explanation of why these solutions are relevant to the query.
        """

        est_input_tokens = len(query) // 4  # Simple estimate
        llm_response = await get_llm_response(prompt, context, model=request.model)
        est_output_tokens = len(llm_response) // 4  # Simple estimate
        total_tokens = est_input_tokens + est_output_tokens

        now = time.time()
        for _ in range(total_tokens):
            token_bucket.append(now)

        return ChatCompletionResponse(
            id=f"chatcmpl-{os.urandom(4).hex()}",
            object="chat.completion",
            created=int(now),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=Message(role="assistant", content=llm_response),
                    finish_reason="stop",
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=est_input_tokens,
                completion_tokens=est_output_tokens,
                total_tokens=total_tokens,
            ),
        )

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/recommend",
    response_model=RecommendationsResponse,
    dependencies=[Depends(check_rate_limit)],
)
async def recommend_solutions(request: Request):
    """Simple API endpoint to get SHL solution recommendations"""
    try:
        body = await request.json()
        query = body.get("query", "")

        if not query:
            raise HTTPException(status_code=400, detail="Query is required")

        est_tokens = len(query) // 4

        now = time.time()
        for _ in range(est_tokens):
            token_bucket.append(now)

        search_results = brain.search_astra(query, limit=10)

        recommendations = []
        if search_results and hasattr(search_results, "source_nodes"):
            seen_names = set()

            for node in search_results.source_nodes:
                meta = node.metadata
                name = meta.get("name", "")

                if name and name not in seen_names:
                    seen_names.add(name)
                    test_type = meta.get("test_type", "")
                    full_test_type = map_test_type_to_full_name(test_type)

                    recommendations.append(
                        {
                            "name": name,
                            "url": meta.get("link", ""),
                            "remote_testing": meta.get("remote_testing", "No"),
                            "adaptive_irt": meta.get("adaptive_irt", "No"),
                            "test_type": full_test_type,
                        }
                    )

        return RecommendationsResponse(recommendations=recommendations[:10])

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/recommend",
    response_model=QueryResponse,
    dependencies=[Depends(check_rate_limit)],
)
async def recommend_endpoint(request: Request):
    """Endpoint to get assessment recommendations based on job description or natural language query"""
    try:
        body = await request.json()
        query = body.get("query", "")

        if not query:
            raise HTTPException(status_code=400, detail="Query is required")

        est_tokens = len(query) // 4
        now = time.time()
        for _ in range(est_tokens):
            token_bucket.append(now)

        search_results = brain.search_astra(query, limit=10)

        recommended_assessments = []
        if search_results and hasattr(search_results, "source_nodes"):
            seen_names = set()

            for node in search_results.source_nodes:
                meta = node.metadata
                name = meta.get("name", "")

                if name and name not in seen_names:
                    seen_names.add(name)
                    test_type = meta.get("test_type", "")
                    test_type_list = [map_test_type_to_full_name(char) for char in test_type if char]
                    
                    completion_time = meta.get("completion_time", "60")
                    try:
                        if isinstance(completion_time, str):
                            if "minute" in completion_time.lower():
                                duration = int(completion_time.split()[0])
                            else:
                                duration = int(completion_time)
                        else:
                            duration = int(completion_time)
                    except (ValueError, IndexError):
                        duration = 60  
                    
                    description = meta.get("description", name)
                    if not description or description == "nan":
                        description = name

                    recommended_assessments.append(
                        AssessmentItem(
                            url=meta.get("link", ""),
                            adaptive_support="Yes" if meta.get("adaptive_irt", "No") == "Yes" else "No",
                            description=description,
                            duration=duration,
                            remote_support="Yes" if meta.get("remote_testing", "No") == "Yes" else "No",
                            test_type=test_type_list or ["Not specified"]
                        )
                    )

        if not recommended_assessments and search_results and hasattr(search_results, "source_nodes") and search_results.source_nodes:
            # Get the first node if we have results but none passed our filtering
            node = search_results.source_nodes[0]
            meta = node.metadata
            test_type = meta.get("test_type", "")
            test_type_list = [map_test_type_to_full_name(char) for char in test_type if char]
            
            completion_time = meta.get("completion_time", "60")
            try:
                if isinstance(completion_time, str):
                    if "minute" in completion_time.lower():
                        duration = int(completion_time.split()[0])
                    else:
                        duration = int(completion_time)
                else:
                    duration = int(completion_time)
            except (ValueError, IndexError):
                duration = 60
            
            description = meta.get("description", "")
            if not description or description == "nan":
                description = meta.get("name", "")

            recommended_assessments.append(
                AssessmentItem(
                    url=meta.get("link", ""),
                    adaptive_support="Yes" if meta.get("adaptive_irt", "No") == "Yes" else "No",
                    description=description,
                    duration=duration,
                    remote_support="Yes" if meta.get("remote_testing", "No") == "Yes" else "No",
                    test_type=test_type_list or ["Not specified"]
                )
            )

        return QueryResponse(recommended_assessments=recommended_assessments[:10])

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
