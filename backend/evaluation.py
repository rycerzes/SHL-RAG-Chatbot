import os
import json
import asyncio
import httpx
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import google.generativeai as genai
import instructor

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import DeepEvalBaseLLM
from deepeval import evaluate

# Load environment variables
load_dotenv()
os.environ["DEEPEVAL_RESULTS_FOLDER"] = "./results"

# Define the golden set with queries and expected relevant assessments
GOLDEN_SET = [
    {
        "query": "I need an assessment for software developers",
        "relevant_assessments": [
            "Coding Pro",
            "Software Developer Test",
            "Cognitive Ability Assessment",
            "Programming Skills Test"
        ]
    },
    {
        "query": "Looking for personality assessments for leadership roles",
        "relevant_assessments": [
            "Leadership Assessment Inventory",
            "Personality Questionnaire",
            "Executive Leadership Assessment",
            "Behavioral Assessment"
        ]
    },
    {
        "query": "Customer service aptitude testing",
        "relevant_assessments": [
            "Customer Service Assessment",
            "Situational Judgment Test",
            "Communication Skills Assessment",
            "Service Orientation Test"
        ]
    },
    {
        "query": "Cognitive ability tests for graduates",
        "relevant_assessments": [
            "Graduate Reasoning Test",
            "Cognitive Ability Assessment",
            "Numerical Reasoning Test",
            "Verbal Reasoning Assessment"
        ]
    },
    {
        "query": "Sales role competency evaluation",
        "relevant_assessments": [
            "Sales Competency Assessment",
            "Motivational Questionnaire",
            "Sales Aptitude Test",
            "Negotiation Skills Assessment"
        ]
    }
]

class CustomGeminiPro(DeepEvalBaseLLM):
    def __init__(self):
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel(model_name="gemini-1.5-pro")

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        client = self.load_model()
        instructor_client = instructor.from_gemini(
            client=client,
            mode=instructor.Mode.GEMINI_JSON,
        )
        resp = instructor_client.messages.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            response_model=schema,
        )
        return resp

    async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        return self.generate(prompt, schema)

    def get_model_name(self):
        return "Gemini 1.5 Pro"

class MetricResult(BaseModel):
    score: float = Field(description="The score for the metric")
    explanation: str = Field(description="Explanation of why this score was given")

class MeanRecallAtK(BaseMetric):
    """Metric for calculating Mean Recall@K across all test cases"""
    
    def __init__(self, k: int = 5, model=None):
        self.k = k
        self.name = f"Mean Recall@{k}"
        self.score_threshold = 0.7
        self.model = model or CustomGeminiPro()
        
    def _compute_recall_at_k(self, recommendations: List[Dict[str, Any]], 
                           relevant_assessments: List[str]) -> float:
        """Compute Recall@K for a single query"""
        if not relevant_assessments:
            return 0.0
            
        # Get the assessment names in the top k recommendations
        rec_names = [rec["name"] for rec in recommendations[:self.k]]
        
        # Count how many relevant assessments were retrieved
        relevant_found = sum(1 for name in rec_names if any(
            relevant.lower() in name.lower() or name.lower() in relevant.lower() 
            for relevant in relevant_assessments
        ))
        
        # Calculate recall
        return relevant_found / len(relevant_assessments)
    
    async def _acompute(self, test_case: LLMTestCase) -> Tuple[float, Dict[str, Any]]:
        query = test_case.input
        relevant_assessments = test_case.expected_output
        actual_recommendations = test_case.actual_output
        
        # Compute the recall score
        score = self._compute_recall_at_k(actual_recommendations, relevant_assessments)
        
        # Use Gemini to provide explanations for the recall score
        prompt = f"""
        Evaluate the recall@{self.k} for the following search query and recommendations:
        
        Query: "{query}"
        
        Relevant assessments (ground truth):
        {json.dumps(relevant_assessments, indent=2)}
        
        Retrieved assessments (top {self.k}):
        {json.dumps([rec['name'] for rec in actual_recommendations[:self.k]], indent=2)}
        
        The recall@{self.k} score is {score:.4f}.
        
        Explain why this score was given and how it could be improved.
        """
        
        result = await self.model.a_generate(prompt, MetricResult)
        
        details = {
            "query": query,
            "relevant_assessments": relevant_assessments,
            "recommendations_retrieved": [rec["name"] for rec in actual_recommendations[:self.k]],
            "k": self.k,
            "recall_score": score,
            "explanation": result.explanation
        }
        
        return score, details

class MeanAveragePrecisionAtK(BaseMetric):
    """Metric for calculating Mean Average Precision@K (MAP@K) across all test cases"""
    
    def __init__(self, k: int = 5, model=None):
        self.k = k
        self.name = f"MAP@{k}"
        self.score_threshold = 0.5
        self.model = model or CustomGeminiPro()
        
    def _compute_ap_at_k(self, recommendations: List[Dict[str, Any]], 
                       relevant_assessments: List[str]) -> float:
        """Compute Average Precision@K for a single query"""
        if not relevant_assessments:
            return 0.0
            
        sum_precision = 0
        num_relevant_found = 0
        
        for i, rec in enumerate(recommendations[:self.k]):
            # Check if current recommendation is relevant
            is_relevant = any(
                relevant.lower() in rec["name"].lower() or rec["name"].lower() in relevant.lower() 
                for relevant in relevant_assessments
            )
            
            if is_relevant:
                num_relevant_found += 1
                # Precision at current position
                precision_at_i = num_relevant_found / (i + 1)
                sum_precision += precision_at_i
        
        # Calculate AP (Average Precision)
        min_value = min(self.k, len(relevant_assessments))
        return sum_precision / min_value if min_value > 0 else 0.0
    
    async def _acompute(self, test_case: LLMTestCase) -> Tuple[float, Dict[str, Any]]:
        query = test_case.input
        relevant_assessments = test_case.expected_output
        actual_recommendations = test_case.actual_output
        
        # Compute the AP score
        score = self._compute_ap_at_k(actual_recommendations, relevant_assessments)
        
        # Use Gemini to provide explanations for the MAP score
        prompt = f"""
        Evaluate the average precision at {self.k} (AP@{self.k}) for the following search query and recommendations:
        
        Query: "{query}"
        
        Relevant assessments (ground truth):
        {json.dumps(relevant_assessments, indent=2)}
        
        Retrieved assessments (in order):
        {json.dumps([rec['name'] for rec in actual_recommendations[:self.k]], indent=2)}
        
        The AP@{self.k} score is {score:.4f}.
        
        Explain why this score was given, focusing on both precision and the ranking order.
        Also suggest how the recommendation quality could be improved.
        """
        
        result = await self.model.a_generate(prompt, MetricResult)
        
        details = {
            "query": query,
            "relevant_assessments": relevant_assessments,
            "recommendations_retrieved": [rec["name"] for rec in actual_recommendations[:self.k]],
            "k": self.k,
            "ap_score": score,
            "explanation": result.explanation
        }
        
        return score, details

async def get_recommendations(query: str) -> List[Dict[str, Any]]:
    """Get recommendations from the API for a given query"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:8000/api/recommend",
                json={"query": query},
                timeout=30.0
            )
            
            if response.status_code != 200:
                print(f"Error: API returned status code {response.status_code}")
                return []
                
            result = response.json()
            return result.get("recommendations", [])
    except Exception as e:
        print(f"Error calling API: {str(e)}")
        return []

async def create_test_cases() -> List[LLMTestCase]:
    """Create test cases from the golden set"""
    test_cases = []
    
    for entry in GOLDEN_SET:
        query = entry["query"]
        relevant_assessments = entry["relevant_assessments"]
        
        # Get actual recommendations from the API
        actual_recommendations = await get_recommendations(query)
        
        # Create a test case
        test_case = LLMTestCase(
            input=query,
            actual_output=actual_recommendations,
            expected_output=relevant_assessments
        )
        
        test_cases.append(test_case)
    
    return test_cases

async def evaluate_system(k_values: List[int] = [3, 5, 10]):
    """Evaluate the recommendation system using various metrics"""
    test_cases = await create_test_cases()
    
    results = {}
    
    for k in k_values:
        print(f"\n--- Evaluating with k={k} ---")
        
        # Create custom model instance
        gemini_model = CustomGeminiPro()
        
        # Run evaluation with Recall@K and MAP@K metrics
        recall_metric = MeanRecallAtK(k=k, model=gemini_model)
        map_metric = MeanAveragePrecisionAtK(k=k, model=gemini_model)
        
        # Use deepeval to evaluate the metrics
        # Use deepeval to evaluate the metrics
        evaluation_results = await evaluate(
            test_cases=test_cases,
            metrics=[recall_metric, map_metric],
            raise_exceptions=False
        )
        # Store results
        results[f"k={k}"] = {
            "recall": evaluation_results[f"Mean Recall@{k}"],
            "map": evaluation_results[f"MAP@{k}"]
        }
        
        # Print summary results
        print(f"Mean Recall@{k}: {evaluation_results[f'Mean Recall@{k}']:.4f}")
        print(f"MAP@{k}: {evaluation_results[f'MAP@{k}']:.4f}")
        
        # Detailed results per query
        print("\nDetailed results per query:")
        for i, test_case in enumerate(test_cases):
            details_recall = recall_metric._compute_recall_at_k(
                test_case.actual_output, test_case.expected_output
            )
            details_map = map_metric._compute_ap_at_k(
                test_case.actual_output, test_case.expected_output
            )
            
            print(f"Query {i+1}: {test_case.input}")
            print(f"  - Recall@{k}: {details_recall:.4f}")
            print(f"  - AP@{k}: {details_map:.4f}")
            print(f"  - Retrieved: {[rec['name'] for rec in test_case.actual_output[:k]]}")
            print(f"  - Relevant: {test_case.expected_output}")
            
    # Save results to a file
    save_path = "./results/evaluation_summary.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nEvaluation summary saved to {save_path}")

if __name__ == "__main__":
    asyncio.run(evaluate_system())