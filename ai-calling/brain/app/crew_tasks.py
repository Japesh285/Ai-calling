# tech-support-service/app/services/crew_tasks.py
# -*- coding: utf-8 -*-

from crewai import Agent, Task, Crew, Process, LLM
from services.api_client import AsyncAPIClient

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from typing import Dict, List, Any
from dotenv import load_dotenv
from langdetect import detect, DetectorFactory
import re
import asyncio
import logging
import json
import os
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import litellm

# warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()

# litellm._turn_on_debug()
# print("OpenAI API Key:", os.getenv("OPENAI_API_KEY"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('support_crew.log'),
        logging.StreamHandler()
    ]
)

# Set seed for reproducibility
DetectorFactory.seed = 0

# Shared API client instance
api_client = AsyncAPIClient()


# ITR Tools
class ITRStatusTool(BaseTool):
    """Tool for fetching ITR status."""
    name: str = "itr_status_tool"
    description: str = "Fetches the status of an ITR transaction."

    class InputSchema(BaseModel):
        txnid: str = Field(..., description="Transaction ID (TXxxxxxxxxxx)")
        token: str = Field(..., description="User authentication token")

    async def _run(self, txnid: str, token: str) -> Dict:
        try:
            return await api_client.get_status(token=token, req_type="ITR", key="status", txnid=txnid)
        except Exception as e:
            return {"error": str(e)}

    def _run(self, txnid: str, token: str) -> Dict:
        """Synchronous wrapper for async operation"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self._run_async(txnid, token))
        except RuntimeError:
            # If no event loop is running, create a new one
            return asyncio.run(self._run_async(txnid, token))
    
    async def _run_async(self, txnid: str, token: str) -> Dict:
        try:
            return await api_client.get_status(token=token, req_type="ITR", key="status", txnid=txnid)
        except Exception as e:
            return {"error": str(e)}


class ITRRefundTool(BaseTool):
    """Tool for fetching ITR refund status."""
    name: str = "itr_refund_tool"
    description: str = "Fetches the refund status of an ITR transaction."

    class InputSchema(BaseModel):
        txnid: str = Field(..., description="Transaction ID (TXxxxxxxxxxx)")
        token: str = Field(..., description="User authentication token")

    async def _run(self, txnid: str, token: str) -> Dict:
        try:
            return await api_client.get_status(token=token, req_type="ITR", key="refund", txnid=txnid)
        except Exception as e:
            return {"error": str(e)}


class ITRPrioritizedTool(BaseTool):
    """Tool for fetching prioritized ITR status."""
    name: str = "itr_prioritized_tool"
    description: str = "Fetches the prioritized status of an ITR transaction."

    class InputSchema(BaseModel):
        txnid: str = Field(..., description="Transaction ID (TXxxxxxxxxxx)")
        token: str = Field(..., description="User authentication token")

    async def _run(self, txnid: str, token: str) -> Dict:
        try:
            return await api_client.get_status(token=token, req_type="ITR", key="priortised", txnid=txnid)
        except Exception as e:
            return {"error": str(e)}


# GST Tools
class GSTReturnStatusTool(BaseTool):
    """Tool for fetching GST return status."""
    name: str = "gst_return_status_tool"
    description: str = "Fetches the status of a GST return transaction."

    class InputSchema(BaseModel):
        txnid: str = Field(..., description="Transaction ID (TXxxxxxxxxxx)")
        token: str = Field(..., description="User authentication token")

    async def _run(self, txnid: str, token: str) -> Dict:
        try:
            return await api_client.get_status(token=token, req_type="GSTRETURN", key="status", txnid=txnid)
        except Exception as e:
            return {"error": str(e)}


class GSTReturnRefundTool(BaseTool):
    """Tool for fetching GST return refund status."""
    name: str = "gst_return_refund_tool"
    description: str = "Fetches the refund status of a GST return transaction."

    class InputSchema(BaseModel):
        txnid: str = Field(..., description="Transaction ID (TXxxxxxxxxxx)")
        token: str = Field(..., description="User authentication token")

    async def _run(self, txnid: str, token: str) -> Dict:
        try:
            return await api_client.get_status(token=token, req_type="GSTRETURN", key="refund", txnid=txnid)
        except Exception as e:
            return {"error": str(e)}


class GSTReturnPrioritizedTool(BaseTool):
    """Tool for fetching prioritized GST return status."""
    name: str = "gst_return_prioritized_tool"
    description: str = "Fetches the prioritized status of a GST return transaction."

    class InputSchema(BaseModel):
        txnid: str = Field(..., description="Transaction ID (TXxxxxxxxxxx)")
        token: str = Field(..., description="User authentication token")

    async def _run(self, txnid: str, token: str) -> Dict:
        try:
            return await api_client.get_status(token=token, req_type="GSTRETURN", key="priortised", txnid=txnid)
        except Exception as e:
            return {"error": str(e)}


# PAN Tools
class PANStatusTool(BaseTool):
    """Tool for fetching PAN status."""
    name: str = "pan_status_tool"
    description: str = "Fetches the status of a PAN transaction."

    class InputSchema(BaseModel):
        txnid: str = Field(..., description="Transaction ID (TXxxxxxxxxxx)")
        token: str = Field(..., description="User authentication token")

    async def _run(self, txnid: str, token: str) -> Dict:
        try:
            return await api_client.get_status(token=token, req_type="PAN", key="status", txnid=txnid)
        except Exception as e:
            return {"error": str(e)}


class PANRefundTool(BaseTool):
    """Tool for fetching PAN refund status."""
    name: str = "pan_refund_tool"
    description: str = "Fetches the refund status of a PAN transaction."

    class InputSchema(BaseModel):
        txnid: str = Field(..., description="Transaction ID (TXxxxxxxxxxx)")
        token: str = Field(..., description="User authentication token")

    async def _run(self, txnid: str, token: str) -> Dict:
        try:
            return await api_client.get_status(token=token, req_type="PAN", key="refund", txnid=txnid)
        except Exception as e:
            return {"error": str(e)}


class PANPrioritizedTool(BaseTool):
    """Tool for fetching prioritized PAN status."""
    name: str = "pan_prioritized_tool"
    description: str = "Fetches the prioritized status of a PAN transaction."

    class InputSchema(BaseModel):
        txnid: str = Field(..., description="Transaction ID (TXxxxxxxxxxx)")
        token: str = Field(..., description="User authentication token")

    async def _run(self, txnid: str, token: str) -> Dict:
        try:
            return await api_client.get_status(token=token, req_type="PAN", key="priortised", txnid=txnid)
        except Exception as e:
            return {"error": str(e)}


# Insurance Tools
class InsuranceStatusTool(BaseTool):
    """Tool for fetching insurance status."""
    name: str = "insurance_status_tool"
    description: str = "Fetches the status of an insurance transaction."

    class InputSchema(BaseModel):
        txnid: str = Field(..., description="Transaction ID (TXxxxxxxxxxx)")
        token: str = Field(..., description="User authentication token")

    async def _run(self, txnid: str, token: str) -> Dict:
        try:
            return await api_client.get_status(token=token, req_type="INSURANCE", key="status", txnid=txnid)
        except Exception as e:
            return {"error": str(e)}


# BCERT Tools
class BCERTStatusTool(BaseTool):
    """Tool for fetching BCERT status."""
    name: str = "bcert_status_tool"
    description: str = "Fetches the status of a BCERT transaction."

    class InputSchema(BaseModel):
        txnid: str = Field(..., description="Transaction ID (TXxxxxxxxxxx)")
        token: str = Field(..., description="User authentication token")

    async def _run(self, txnid: str, token: str) -> Dict:
        try:
            return await api_client.get_status(token=token, req_type="BCERT", key="status", txnid=txnid)
        except Exception as e:
            return {"error": str(e)}


class BCERTRefundTool(BaseTool):
    """Tool for fetching BCERT refund status."""
    name: str = "bcert_refund_tool"
    description: str = "Fetches the refund status of a BCERT transaction."

    class InputSchema(BaseModel):
        txnid: str = Field(..., description="Transaction ID (TXxxxxxxxxxx)")
        token: str = Field(..., description="User authentication token")

    async def _run(self, txnid: str, token: str) -> Dict:
        try:
            return await api_client.get_status(token=token, req_type="BCERT", key="refund", txnid=txnid)
        except Exception as e:
            return {"error": str(e)}


class BCERTPrioritizedTool(BaseTool):
    """Tool for fetching prioritized BCERT status."""
    name: str = "bcert_prioritized_tool"
    description: str = "Fetches the prioritized status of a BCERT transaction."

    class InputSchema(BaseModel):
        txnid: str = Field(..., description="Transaction ID (TXxxxxxxxxxx)")
        token: str = Field(..., description="User authentication token")

    async def _run(self, txnid: str, token: str) -> Dict:
        try:
            return await api_client.get_status(token=token, req_type="BCERT", key="priortised", txnid=txnid)
        except Exception as e:
            return {"error": str(e)}


# BSPR Tools
class BSPRStatusTool(BaseTool):
    """Tool for fetching BSPR status."""
    name: str = "bspr_status_tool"
    description: str = "Fetches the status of a BSPR transaction."

    class InputSchema(BaseModel):
        txnid: str = Field(..., description="Transaction ID (TXxxxxxxxxxx)")
        token: str = Field(..., description="User authentication token")

    async def _run(self, txnid: str, token: str) -> Dict:
        try:
            return await api_client.get_status(token=token, req_type="BSPR", key="status", txnid=txnid)
        except Exception as e:
            return {"error": str(e)}


class BSPRRefundTool(BaseTool):
    """Tool for fetching BSPR refund status."""
    name: str = "bspr_refund_tool"
    description: str = "Fetches the refund status of a BSPR transaction."

    class InputSchema(BaseModel):
        txnid: str = Field(..., description="Transaction ID (TXxxxxxxxxxx)")
        token: str = Field(..., description="User authentication token")

    async def _run(self, txnid: str, token: str) -> Dict:
        try:
            return await api_client.get_status(token=token, req_type="BSPR", key="refund", txnid=txnid)
        except Exception as e:
            return {"error": str(e)}


class BSPRPrioritizedTool(BaseTool):
    """Tool for fetching prioritized BSPR status."""
    name: str = "bspr_prioritized_tool"
    description: str = "Fetches the prioritized status of a BSPR transaction."

    class InputSchema(BaseModel):
        txnid: str = Field(..., description="Transaction ID (TXxxxxxxxxxx)")
        token: str = Field(..., description="User authentication token")

    async def _run(self, txnid: str, token: str) -> Dict:
        try:
            return await api_client.get_status(token=token, req_type="BSPR", key="priortised", txnid=txnid)
        except Exception as e:
            return {"error": str(e)}


# GSTREG Tools
class GSTREGStatusTool(BaseTool):
    """Tool for fetching GSTREG status."""
    name: str = "gstreg_status_tool"
    description: str = "Fetches the status of a GSTREG transaction."

    class InputSchema(BaseModel):
        txnid: str = Field(..., description="Transaction ID (TXxxxxxxxxxx)")
        token: str = Field(..., description="User authentication token")

    async def _run(self, txnid: str, token: str) -> Dict:
        try:
            return await api_client.get_status(token=token, req_type="GSTREG", key="status", txnid=txnid)
        except Exception as e:
            return {"error": str(e)}


class GSTREGRefundTool(BaseTool):
    """Tool for fetching GSTREG refund status."""
    name: str = "gstreg_refund_tool"
    description: str = "Fetches the refund status of a GSTREG transaction."

    class InputSchema(BaseModel):
        txnid: str = Field(..., description="Transaction ID (TXxxxxxxxxxx)")
        token: str = Field(..., description="User authentication token")

    async def _run(self, txnid: str, token: str) -> Dict:
        try:
            return await api_client.get_status(token=token, req_type="GSTREG", key="refund", txnid=txnid)
        except Exception as e:
            return {"error": str(e)}


class GSTREGPrioritizedTool(BaseTool):
    """Tool for fetching prioritized GSTREG status."""
    name: str = "gstreg_prioritized_tool"
    description: str = "Fetches the prioritized status of a GSTREG transaction."

    class InputSchema(BaseModel):
        txnid: str = Field(..., description="Transaction ID (TXxxxxxxxxxx)")
        token: str = Field(..., description="User authentication token")

    async def _run(self, txnid: str, token: str) -> Dict:
        try:
            return await api_client.get_status(token=token, req_type="GSTREG", key="priortised", txnid=txnid)
        except Exception as e:
            return {"error": str(e)}



# class CallbackSchedulerTool(BaseTool):
#     """Tool for scheduling callbacks."""
#     name: str = "callback_scheduler"
#     description: str = "Schedule callbacks with time optimization"

#     class InputSchema(BaseModel):
#         name: str = Field(..., description="User's full name")
#         mobile: str = Field(..., description="10-digit mobile number")
#         preferred_time: str = Field(..., description="Preferred callback time")

#     def _run(self, name: str, mobile: str, preferred_time: str) -> str:
#         """Schedules a callback."""
#         if not all([name, mobile, preferred_time]):
#             return "Please provide name, mobile, and preferred callback time"
#         return f"Callback scheduled for {preferred_time}. We'll contact {name} on {mobile}"


class ClassifierTool(BaseTool):
    """Tool for classifying queries into predefined categories using department name."""
    name: str = "classifier_tool"
    description: str = "Classifies user queries based on department-specific criteria."

    _candidate_labels: List[str] = PrivateAttr()
    _department_name: str = PrivateAttr()
    _llm: LLM = PrivateAttr()

    def __init__(self, llm: LLM, department_name: str, **kwargs):
        super().__init__(**kwargs)
        self._llm = llm
        self._department_name = department_name
        self._candidate_labels = [
            "GST-STATUS", "ITR-STATUS", "INSURANCE-POLICY-STATUS", "BUSINESS-CERT-STATUS",
            "CALLBACK", "PAN-GENERAL", "PAN-STATUS", "ITR-REFUND", "GENERAL"
        ]

    def _run(self, query: str) -> str:
        """Classifies the query and returns the category."""
        query = query.strip().lower()
        
        # Token limitation
        max_tokens = 50
        if len(query.split()) > max_tokens:
            return "Query exceeds the maximum token limit."

        prompt = (
            f"You are a highly skilled query classifier. Your task is to classify the following user query into one of these categories: "
            f"{', '.join(self._candidate_labels)}. "
            f"Return only one category label exactly as it appears in the list above. "
            f"Here are some examples:\n"
            f"1. 'What is the status of my GST registration?' -> 'GST-STATUS'\n"
            f"2. 'I need to know my ITR refund status.' -> 'ITR-STATUS'\n"
            f"3. 'Can you help me with my insurance policy?' -> 'INSURANCE-POLICY-STATUS'\n"
            f"4. 'How can I schedule a callback?' -> 'CALLBACK'\n"
            f"5. 'What are the current tax slabs?' -> 'GENERAL'\n"
            f"Query: \"{query}\"\n"
            f"Category:"
        )
        
        try:
            response = self._llm.call(prompt=prompt, temperature=0.2, max_tokens=20)
            result = response.strip()

            if result in self._candidate_labels:
                return result
            else:
                return "GENERAL"
        except Exception as e:
            logging.error(f"Error during classification: {e}")
            return "GENERAL"


class GeneralInfoTool(BaseTool):
    """Tool for providing legal and financial information about Indian taxation and insurance."""
    
    name: str = "general_info_tool"
    description: str = "Provides legal and financial information about Indian taxation and insurance."

    class InputSchema(BaseModel):
        query: str = Field(..., description="User query for information")

    _llm: LLM = PrivateAttr()

    def __init__(self, llm: LLM, **kwargs):
        super().__init__(**kwargs)
        self._llm = llm

    def _run(self, query: str) -> str:
        """Handles the user query and provides relevant information."""
        
        # Token limitation
        max_tokens = 50
        if len(query.split()) > max_tokens:
            return "Query exceeds the maximum token limit."

        prompt = (
            f"You are an expert in Indian taxation and insurance. "
            f"Answer the following query with precise and accurate information: \"{query}\""
        )
        try:
            response = self._llm.call(messages=prompt)
            # Simulate confidence score
            confidence_score = 0.8  # Replace with actual confidence score calculation
            return {"answer": response.strip(), "confidence": confidence_score}
        except Exception as e:
            logging.error(f"Error during information retrieval: {e}")
            return "I'm sorry, I couldn't retrieve the information at this time."


# Load the multilingual model and tokenizer
model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Define Language Detection Tool
class LanguageDetectionTool(BaseTool):
    """Tool for detecting the language of a user query using LLM."""
    name: str = "language_detection_tool"
    description: str = "Detects the language of the user query."

    _llm: LLM = PrivateAttr()

    def __init__(self, llm: LLM, **kwargs):
        super().__init__(**kwargs)
        self._llm = llm

    class InputSchema(BaseModel):
        query: str = Field(..., description="User query for language detection")

    async def _run(self, query: str) -> str:
        """Detects the language of the query using the LLM."""
        prompt = (
            "You are a language detection expert. Analyze the following text and determine its language.\n"
            "Rules:\n"
            "- Hindi: Text in Devanagari script or pure Hindi vocabulary\n"
            "- Hinglish: Mix of Hindi words in Roman script with English words\n"
            "- English: Pure English text\n\n"
            "Examples:\n"
            "- 'मैं आईटीआर कब भर सकता हूं?' → Hindi\n"
            "- 'Mein ITR kab bhar sakta hun?' → Hinglish\n"
            "- 'When can I file my ITR?' → English\n\n"
            f"Text: {query}\n"
            "Language (respond with only 'Hindi', 'Hinglish', or 'English'):"
        )
        
        logging.info(f"Prompt sent to LLM: {prompt}")

        try:
            response = await asyncio.wait_for(
                self._llm.call(prompt=prompt, temperature=0.2, max_tokens=10),
                timeout=5.0
            )
        except asyncio.TimeoutError:
            logging.error("Language detection timed out.")
            return "Language detection timed out."
        except Exception as e:
            logging.error(f"Error during language detection: {e}")
            return "Error in language detection."

        detected_language = response.strip().lower()
        logging.info(f"Detected language: {detected_language}")

        if detected_language in ['hindi', 'english', 'hinglish']:
            return detected_language
        else:
            logging.warning(f"Unexpected language detected: {detected_language}. Defaulting to English.")
            return 'english'

# Define Support Crew
class SupportCrew:
    """CrewAI crew with specialized agents for handling support queries."""

    def __init__(self, user_data: dict):
        """Initializes the SupportCrew with user data."""
        self.user_data = user_data
        self.llm = LLM(model="openai/gpt-4o", temperature=0.2, max_tokens=500,base_url="https://api.openai.com/v1" ,api_key=os.getenv("OPENAI_API_KEY"))
        self.memory = {"conversation_history": []}
        self.language_detector = self._create_lang_detection_agent()
        self.specialist = self._create_specialist()
        self.quality = self._create_quality_assurance_specialist()
        self.crew = self._create_crew()
        self.index_file_path = 'faiss_index.index'
        self.vector_index = self._load_index()
        self.embeddings = []
        self.threshold = 0.95  # Set the relevancy threshold to 95%

    def _add_to_memory(self, role: str, message: str):
        """Adds a message to the conversation history."""
        self.memory["conversation_history"].append({"role": role, "message": message})

    def _create_specialist(self) -> Agent:
        """Creates a specialist agent dynamically based on the user's department."""
        department_name = self.user_data.get("department_details", {}).get("name", "General")
        department_description = self.user_data.get("department_details", {}).get("description", "Handles general queries")
        user_name = self.user_data.get('name')

        tools = [
        
        GeneralInfoTool(self.llm),
        # ITR
        ITRStatusTool(),
        ITRRefundTool(),
        ITRPrioritizedTool(),
        # GSTRETURN
        GSTReturnStatusTool(),
        GSTReturnRefundTool(),
        GSTReturnPrioritizedTool(),
        # GSTREG
        GSTREGStatusTool(),
        GSTREGRefundTool(),
        GSTREGPrioritizedTool(),
        # PAN
        PANStatusTool(),
        PANRefundTool(),
        PANPrioritizedTool(),
        # INSURANCE
        InsuranceStatusTool(),
        # BSPR
        BSPRStatusTool(),
        BSPRRefundTool(),
        BSPRPrioritizedTool(),
        # BCERT
        BCERTStatusTool(),
        BCERTRefundTool(),
        BCERTPrioritizedTool(),
    ]

        return Agent(
            role=f"Senior Specialist Support Representative",
            goal=f"Be the most friendly and helpful support representative in your team .Provide expert support for {department_name} related queries.",
            backstory=(
                f"You work at FSK (https://fskindia.com/) in {department_name} department and "
                f" are now working on providing "
                f"support to {user_name}, a super important customer "
                f"make sure to use the language detection tool and use the detected language as the response language"
                f" for your company."
                f"You need to make sure that you provide the best support!"
                f"Make sure to provide full complete answers, "
                f"and make no assumptions."
            ),
            tools=tools,
            llm=self.llm,
            max_iter=4,
            max_execution_time=50,
            allow_delegation=False,
            verbose=False
        )

    def _create_lang_detection_agent(self):
       return Agent(
        role="Language Classifier",
        goal="Classify the input query into one of the following languages: Hindi, English, or Hinglish. Respond with exactly one word: 'Hindi', 'English', or 'Hinglish'.",
        backstory=(
            "You are a highly skilled language detection expert trained to identify whether a given query is written in Hindi, English, or Hinglish (Hindi written using English/Roman script). "
            "You are assisting a support system and your sole job is to classify the language of the incoming user query. "
            "You take into account both the script (Devanagari or Roman) and the grammatical structure of the sentence. "
            "If a query is written in English alphabet but clearly follows Hindi grammar and context like 'meri ITR ka status kya hai' or 'kab bhar sakta hoon'you must classify it as Hinglish, not Hindi. "
            "If the query is written in Devanagari script (e.g., 'मेरा नाम क्या है?'), classify it as Hindi. "
            "If the sentence follows English grammar and vocabulary, classify it as English. "
            "Always return just one word from the set: 'Hindi', 'English', or 'Hinglish'."
        ),
        verbose=False,
        allow_delegation=False,
        
    )

    
    def _create_quality_assurance_specialist(self):
        """Creates a quality assurance specialist agent."""
        department_details = self.user_data.get("department_details", {})
        department_name = department_details.get("name", "General")
        user_name = self.user_data.get('name')

        return Agent(
            role=f"{department_name.upper()} SUPPORT QUALITY ASSURANCE SPECIALIST",
            goal="Get recognition for providing the best support quality assurance in your team",
            backstory=(
                f"You work at FSK(https://fskindia.com/) in {department_name} department and "
                f"are now working with your team "
                f"on a request from {user_name} ensuring that "
                f"the support representative is providing the best support possible.\n"
                f"You need to make sure that the support representative is providing full "
                f"complete answers, and make no assumptions.\n"
                f"If users ask to join as a retailer or are willing to signup, ensure they receive proper guidance."
                
            ),
            verbose=False,
            allow_delegation=False,
        )

 
    def _create_crew(self) -> Crew:
        """Creates the crew with the specialist, classifier, and routing agents."""
        lang_detection = Task(
            description=(
                "{name} just reached out with a super important ask:\n"
                "{query}\n\n"
                "Make sure to use everything you know "
                "to detect the language of the query"
            ),
            expected_output="name of the language like hindi english or hinglish",     
            agent=self.language_detector
        )

        tools = [
        
        GeneralInfoTool(self.llm),
        # ITR
        ITRStatusTool(),
        ITRRefundTool(),
        ITRPrioritizedTool(),
        # GSTRETURN
        GSTReturnStatusTool(),
        GSTReturnRefundTool(),
        GSTReturnPrioritizedTool(),
        # GSTREG
        GSTREGStatusTool(),
        GSTREGRefundTool(),
        GSTREGPrioritizedTool(),
        # PAN
        PANStatusTool(),
        PANRefundTool(),
        PANPrioritizedTool(),
        # INSURANCE
        InsuranceStatusTool(),
        # BSPR
        BSPRStatusTool(),
        BSPRRefundTool(),
        BSPRPrioritizedTool(),
        # BCERT
        BCERTStatusTool(),
        BCERTRefundTool(),
        BCERTPrioritizedTool(),
    ]



        inquiry_task = Task(
            description=(
                "{name} just reached out with a super important ask:\n"
                "{query}\n\n"
                "Make sure to use everything you know "
                "If the query contains a transaction ID (e.g., TX1234567890), extract it as 'txnid'. "
                "Use the user's {token} from the provided user data as 'token'. "
                "Pass both 'txnid' and 'token' to the appropriate tool for status, refund, or prioritized queries."
                "Make sure to use the correct tool for the transaction type."
                "to provide the best support possible."
                "You must strive to provide a complete "
                "and accurate response to the customer's inquiry."
                "must use the language detected in language detected task to answer use queries"
                "In case user asks some irrelavant question or query that is not related to the tools you have, just reply in short that you cannot help with that query can help with queries related to services by FSK India.\n"
                "and if general greeting is detected, just reply with a friendly greeting and ask them how you can help them today"
            ),
            expected_output=(
                "A summerized, informative response to the "
                "customer's inquiry that addresses "
                "all aspects of their question.\n"
                "The response should include references "
                "to everything you used to find the answer, "
                "including external data or solutions. "
                "Ensure the answer is complete, "
                "leaving no questions unanswered, and maintain a helpful and friendly "
                "tone throughout."
            ),
            tools=tools,
            agent=self.specialist,
            context=[lang_detection]
        )
        
        quality_assurance_review = Task(
            description=(
                "You work at FSK India (https://fskindia.com) and "
                "are now working with your team "
                "on a request from {name} ensuring that "
                "the support representative is "
                "providing the best support possible.\n"
                "You need to make sure that the support representative "
                "is providing full"
                "complete answers, and make no assumptions."
         
            ),
            expected_output=(
                "A final, summarized, and informative response "
                "ready to be sent to the customer.\n"
                "This response should fully address the "
                "customer's inquiry in the language as detected by lang_detection task, incorporating all "
                "relevant feedback and improvements.\n"
                "Don't be too formal, we are a chill and cool company "
                "but maintain a professional and friendly tone throughout."
                "make sure it's in the language the customer has asked the query in"
            ),
            agent=self.quality,
            
            
        )
                
        return Crew(
            agents=[self.language_detector,self.specialist, self.quality],
            tasks=[lang_detection,inquiry_task, quality_assurance_review],
            process=Process.sequential, 
            verbose=False,
        )

 
    def _load_index(self):
        """Loads the FAISS index from a file or creates a new one if it doesn't exist."""
        if os.path.exists(self.index_file_path):
            return faiss.read_index(self.index_file_path)
        else:
            embedding_dimension = 768
            return faiss.IndexFlatL2(embedding_dimension)

    def _save_index(self):
        """Saves the FAISS index to a file."""
        faiss.write_index(self.vector_index, self.index_file_path)

    def _get_embedding(self, query: str):
        """Generates an embedding for the query using a multilingual transformer model."""
        inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).numpy()
        if embedding.shape[1] != 768:
            raise ValueError(f"Embedding dimension mismatch: expected 768, got {embedding.shape[1]}")
        return embedding

    def _retrieve_similar_queries(self, query: str, k: int = 5):
        """Retrieves the top k similar queries from the FAISS index."""
        embedding = self._get_embedding(query)
        D, I = self.vector_index.search(embedding, k)
        return D, I

    async def handle_query(self, query: str) -> str:
        """Handles user queries by orchestrating the crew asynchronously."""
        logging.info("Handling query: %s", query)
        try:
            # Check if the query is transaction-related and contains a transaction ID
            if re.search(r'TX\d{10}', query):
                logging.info("Transaction ID detected. Bypassing vector database check.")
                # Proceed directly with the crew
                embedding = self._get_embedding(query)
                self.vector_index.add(embedding)
                self.embeddings.append(query)
                self._save_index()

                result = await self.crew.kickoff_async({"query": query, **self.user_data})
                self._add_to_memory(role="System", message=result)
                logging.info("Result from crew: %s", result)
                return result
            else:
                # Retrieve similar queries from the vector database
                distances, indices = self._retrieve_similar_queries(query, k=1)

                # Check if the most relevant query exceeds the relevancy threshold
                if distances[0][0] <= (1 - self.threshold) and len(self.embeddings) > 0 and indices[0][0] < len(self.embeddings):
                    # Use the retrieved query as the response
                    retrieved_query_index = indices[0][0]
                    retrieved_query = self.embeddings[retrieved_query_index]
                    logging.info(f"Using retrieved query: {retrieved_query} with distance: {distances[0][0]}")
                    return f"I found a similar query in my database: {retrieved_query}"
                else:
                    # If no relevant query is found, proceed with the crew
                    logging.info("No relevant query found in the database. Proceeding with the crew.")

                    # Store the query embedding in FAISS
                    embedding = self._get_embedding(query)
                    self.vector_index.add(embedding)
                    self.embeddings.append(query)
                    self._save_index()

                    # Kickoff the crew and get the result
                    result = await self.crew.kickoff_async({"query": query, **self.user_data})
                    self._add_to_memory(role="System", message=result)
                    print(f"Result from crew: {result}")
                    print(f"Type:{type(result)}")
                    logging.info("Result from crew: %s", result)
                    if result.json_dict:
                        return json.dumps(result.json_dict, indent=2)
                    elif result.pydantic:  
                        return  result.pydantic
                    else:
                        return result.raw

        except Exception as e:
     
        
            error_message = "I'm sorry, I couldn't understand your query. Could you please provide more details?"
            self._add_to_memory(role="System", message=error_message)
            logging.error("Error in handle_query: %s", e)
            return error_message

    async def handle_batch_queries(self, queries: List[str]) -> List[str]:
        """Handles a batch of user queries by orchestrating the crew asynchronously."""
        results = []
        for query in queries:
            self._add_to_memory(role="User", message=query)
            try:
                result = await self.handle_query(query)
                self._add_to_memory(role="System", message=result)
                results.append(result)
            except Exception as e:
                raise e
                error_message = "I'm sorry, I couldn't understand your query. Could you please provide more details?"
                self._add_to_memory(role="System", message=error_message)
                results.append(error_message)
        return results

# Example user data
# user_data = {
#     "name": "Sandeep Singh",
#     "email": "sundeepsingh.ca@gmail.com",
#     "mobile": "9086025119",
#     "department_id": 4,
#     "token":"fasfafaqtewyeyregweg",
#     "issue_description": "I need to know the current tax slab",
#     "priority": "medium",
#     "department_details": {
#         "name": "taxation",
#         "description": "Handles all taxation-related queries",
#         "manager": None,
#         "users": None,
#         "id": 4,
#         "manager_id": 21
#     },
#     "ticket_number": "9262ca00-ddd8-4ca4-9771-4fe903f98659",
#     "agent_email": "sundeepsingh.cagf@gmail.com"
# }

# # Initialize the SupportCrew
# support_crew = SupportCrew(user_data=user_data)

# # Test a batch of queries
# async def test_batch_queries():
#     queries = [
#         # "What are the tax redemptions offered to senior citizens?",
#         # "What is the status of my transaction TX1234567890?",
#         # "Can you provide information on GST registration?",
#         # "What are the current tax slabs for individuals?",
#         # "How can I schedule a callback for my query?",
#         # "what are the services you provide?",
#         # "whats the link to join you as partner?",
#         # "meri insurance kaa status kya hai?"
#         # "transaction id hai TX1234567890",
#         "who won the match today?",
    
#     ]
#     responses = await support_crew.handle_batch_queries(queries)
#     for response in responses:
#         print(response)

# # Run the test
# asyncio.run(test_batch_queries())

