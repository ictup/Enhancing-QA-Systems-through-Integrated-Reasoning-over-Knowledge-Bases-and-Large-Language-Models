from dotenv import load_dotenv
load_dotenv()  # 加载 .env 文件内容到 os.environ
import logging
logging.getLogger().setLevel(logging.WARNING)
import asyncio # Add this import
from openai import OpenAI, AsyncOpenAI # Ensure AsyncOpenAI is imported
# ... other imports
# 替代聊天模型导入
from langchain_openai import ChatOpenAI
import pandas as pd
# 替代 PromptTemplate 导入
from langchain_core.prompts import PromptTemplate

# 替代 LLMChain 导入
from langchain.chains import LLMChain
import asyncio
import re
import time
import random
from typing import List, Dict, Any, Callable, Tuple, Optional, Union # Make sure Union is imported
import numpy as np
from openai import AsyncOpenAI
import os
# 替代 LLM（如 OpenAI）导入
from langchain_community.llms import OpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
import numpy as np
import re
import string
import neo4j
from neo4j import GraphDatabase, basic_auth
import pandas as pd
from collections import deque
import itertools
from typing import Dict, List
import pickle
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize 
import openai
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor
import os
from PIL import Image, ImageDraw, ImageFont
import csv
from gensim import corpora
from gensim.models import TfidfModel
from gensim.similarities import SparseMatrixSimilarity
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import sys
from time import sleep


def chat_4o_mini(prompt):
    completion = client.chat.completions.create(
        model="gpt-4.1-2025-04-14",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        top_p=0.85,
        max_tokens=1024,
    )
    return completion.choices[0].message.content


def chat_4(prompt):
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        top_p=0.85,
        max_tokens=1024,
    )
    return completion.choices[0].message.content


def zero_shot_prompt_template(question):
    """Enhanced Zero-shot Prompting with structured medical reasoning"""
    return f"""You are a clinical diagnostician. Analyze the patient's presentation using systematic medical reasoning.

PATIENT PRESENTATION: {question}

Please provide a structured medical assessment following this framework:

1. SYMPTOM ANALYSIS: Identify and categorize key symptoms
2. DIFFERENTIAL DIAGNOSIS: List 3-4 most likely conditions with brief rationale
3. DIAGNOSTIC TESTS: Recommend specific tests to confirm diagnosis
4. TREATMENT APPROACH: Suggest evidence-based interventions
5. URGENCY LEVEL: Assess need for immediate vs routine care

Provide concise, clinically relevant recommendations.
If some steps lack sufficient information, clearly state that and avoid assumptions. """

def zero_shot_prompt_template_with_kg(question, response_of_KG_list_path, response_of_KG_neighbor):
    """Enhanced Zero-shot Prompting with structured medical reasoning and KG context."""
    
    kg_path_context = "No specific path-based knowledge graph evidence provided."
    if response_of_KG_list_path and response_of_KG_list_path.strip() and response_of_KG_list_path != '{}':
        kg_path_context = f"""The following path-based evidence from a knowledge graph might be relevant:
<KNOWLEDGE_GRAPH_PATH_EVIDENCE>
{response_of_KG_list_path}
</KNOWLEDGE_GRAPH_PATH_EVIDENCE>"""

    kg_neighbor_context = "No specific neighbor-based knowledge graph evidence provided."
    if response_of_KG_neighbor and response_of_KG_neighbor.strip() and response_of_KG_neighbor != '{}':
        kg_neighbor_context = f"""The following neighbor-based evidence from a knowledge graph might be relevant:
<KNOWLEDGE_GRAPH_NEIGHBOR_EVIDENCE>
{response_of_KG_neighbor}
</KNOWLEDGE_GRAPH_NEIGHBOR_EVIDENCE>"""

    return f"""You are a clinical diagnostician. Analyze the patient's presentation using systematic medical reasoning and any provided knowledge graph context.

PATIENT PRESENTATION: {question}

{kg_path_context}

{kg_neighbor_context}

Please provide a structured medical assessment following this framework. If knowledge graph evidence is provided, integrate it into your reasoning where appropriate, explicitly mentioning how it supports or influences your assessment for each step:

1. SYMPTOM ANALYSIS: Identify and categorize key symptoms. (If KG evidence is relevant here, explain how).
2. DIFFERENTIAL DIAGNOSIS: List 3-4 most likely conditions with brief rationale. (Explicitly state if/how KG evidence supports these diagnoses).
3. DIAGNOSTIC TESTS: Recommend specific tests to confirm diagnosis. (Explain if KG evidence suggests or supports certain tests).
4. TREATMENT APPROACH: Suggest evidence-based interventions. (Mention if KG evidence provides insights into treatments).
5. URGENCY LEVEL: Assess need for immediate vs routine care. (Consider if KG evidence impacts urgency).

Provide concise, clinically relevant recommendations.
If some steps lack sufficient information, clearly state that and avoid assumptions. If KG evidence is present but not relevant to a particular step, you can state that.
When referencing KG evidence, you can refer to it as "Path-based Evidence" or "Neighbor-based Evidence".
"""

def one_shot_prompt_template(question):
    """Improved One-shot with high-quality medical example"""
    return f"""You are a clinical diagnostician. Analyze symptoms systematically to provide accurate medical recommendations.

EXAMPLE CASE:
Patient: "I've had crushing chest pain for 30 minutes, radiating to my left arm, with nausea and sweating."

Medical Analysis:
1. SYMPTOMS: Central crushing chest pain, radiation pattern, autonomic symptoms
2. DIFFERENTIAL: Acute MI (high probability), unstable angina, aortic dissection, pulmonary embolism
3. DIAGNOSTIC TESTS: Immediate ECG, troponins, chest X-ray, D-dimer if PE suspected
4. TREATMENT: Aspirin, nitrates if BP stable, prepare for cardiac catheterization
5. URGENCY: STAT - activate cardiac emergency protocol

NOW ANALYZE THIS CASE:
Patient: "{question}"

Medical Analysis:
1. SYMPTOMS:
2. DIFFERENTIAL:
3. DIAGNOSTIC TESTS:
4. TREATMENT:
5. URGENCY:
If some steps lack sufficient information, clearly state that and avoid assumptions."""


def few_shot_prompt_template(question):
    """Fixed Few-shot with diverse, high-quality examples"""
    return f"""You are a clinical diagnostician. Use systematic medical reasoning to analyze patient presentations.

TRAINING EXAMPLES:

CASE 1:
Patient: "Progressive fatigue over 3 months, pale skin, shortness of breath with minimal exertion."
Analysis:
- SYMPTOMS: Chronic fatigue, pallor, exertional dyspnea - classic anemia triad
- DIFFERENTIAL: Iron deficiency anemia (most common), B12/folate deficiency, chronic disease anemia
- TESTS: CBC with differential, iron studies (ferritin, TIBC, transferrin saturation), B12/folate levels
- TREATMENT: Identify underlying cause, iron supplementation if iron deficient
- URGENCY: Routine unless severe symptoms

CASE 2:
Patient: "Severe headache with neck stiffness, fever, and sensitivity to light for 6 hours."
Analysis:
- SYMPTOMS: Headache, neck rigidity, photophobia, fever - meningeal irritation signs
- DIFFERENTIAL: Bacterial meningitis (emergency), viral meningitis, subarachnoid hemorrhage
- TESTS: STAT lumbar puncture after CT head, blood cultures, PCR studies
- TREATMENT: Empirical antibiotics immediately, supportive care
- URGENCY: CRITICAL - medical emergency

CASE 3:
Patient: "Polyuria, polydipsia, polyphagia, and 15-pound weight loss over 2 months."
Analysis:
- SYMPTOMS: Classic diabetes triad plus weight loss - hyperglycemic syndrome
- DIFFERENTIAL: Type 1 diabetes, Type 2 diabetes, secondary diabetes, DKA risk
- TESTS: Random glucose, HbA1c, urine ketones, arterial blood gas if unwell
- TREATMENT: Insulin if Type 1 or severe, metformin consideration, lifestyle counseling
- URGENCY: Urgent - risk of diabetic complications

NOW ANALYZE:
Patient: "{question}"
Analysis:
- SYMPTOMS:
- DIFFERENTIAL:
- TESTS:
- TREATMENT:
- URGENCY:
If some steps lack sufficient information, clearly state that and avoid assumptions. """


def chain_of_thought_prompt_template(question):
    """Enhanced Chain-of-Thought with medical reasoning structure"""
    return f"""You are a clinical diagnostician. Work through this case step-by-step using systematic medical reasoning.

PATIENT PRESENTATION: "{question}"

STEP 1 - SYMPTOM IDENTIFICATION AND ANALYSIS:
[Identify each symptom and its clinical significance]

STEP 2 - PATHOPHYSIOLOGICAL REASONING:
[Consider what underlying processes could cause these symptoms]

STEP 3 - DIFFERENTIAL DIAGNOSIS FORMATION:
[List possible diagnoses ranked by likelihood and severity]

STEP 4 - DIAGNOSTIC STRATEGY:
[Determine which tests would best differentiate between your top diagnoses]

STEP 5 - TREATMENT PRIORITIZATION:
[Consider immediate vs long-term management needs]

STEP 6 - CLINICAL DECISION SUMMARY:
Medical Recommendation: [Provide final integrated assessment with specific actionable steps]

Please work through each step systematically before providing your final recommendation.
If some steps lack sufficient information, clearly state that and avoid assumptions."""


def prompt_extract_keyword(input_text):
    template = """
    There are some samples:
    \n\n
    ### Instruction:\n'Learn to extract entities from the following medical questions.'\n\n### Input:\n
    <CLS>Doctor, I have been having discomfort and dryness in my vagina for a while now. I also experience pain during sex. What could be the problem and what tests do I need?<SEP>The extracted entities are\n\n ### Output:
    <CLS>Doctor, I have been having discomfort and dryness in my vagina for a while now. I also experience pain during sex. What could be the problem and what tests do I need?<SEP>The extracted entities are Vaginal pain, Vaginal dryness, Pain during intercourse<EOS>
    \n\n
    Instruction:\n'Learn to extract entities from the following medical answers.'\n\n### Input:\n
    <CLS>Okay, based on your symptoms, we need to perform some diagnostic procedures to confirm the diagnosis. We may need to do a CAT scan of your head and an Influenzavirus antibody assay to rule out any other conditions. Additionally, we may need to evaluate you further and consider other respiratory therapy or physical therapy exercises to help you feel better.<SEP>The extracted entities are\n\n ### Output:
    <CLS>Okay, based on your symptoms, we need to perform some diagnostic procedures to confirm the diagnosis. We may need to do a CAT scan of your head and an Influenzavirus antibody assay to rule out any other conditions. Additionally, we may need to evaluate you further and consider other respiratory therapy or physical therapy exercises to help you feel better.<SEP>The extracted entities are CAT scan of head (Head ct), Influenzavirus antibody assay, Physical therapy exercises; manipulation; and other procedures, Other respiratory therapy<EOS>
    \n\n
    Try to output:
    ### Instruction:\n'Learn to extract entities from the following medical questions.'\n\n### Input:\n
    <CLS>{input}<SEP>The extracted entities are\n\n ### Output:
    """

    prompt = PromptTemplate(
        template = template,
        input_variables = ["input"]
    )

    system_message_prompt = SystemMessagePromptTemplate(prompt = prompt)
    system_message_prompt.format(input = input_text)

    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
    chat_prompt_with_values = chat_prompt.format_prompt(input = input_text,\
                                                        text={})

    response_of_KG = chat.invoke(chat_prompt_with_values.to_messages()).content

    question_kg = re.findall(re1,response_of_KG)
    return question_kg



def find_shortest_path(start_entity_name, end_entity_name,candidate_list):
    global exist_entity
    with driver.session() as session:
        result = session.run(
            # "MATCH (start_entity:Entity{name:$start_entity_name}), (end_entity:Entity{name:$end_entity_name}) "
            # "MATCH p = allShortestPaths((start_entity)-[*..5]->(end_entity)) "
            # "RETURN p",
            "MATCH p = allShortestPaths((start:Entity{name:$start_entity_name})-[*..5]->(end:Entity{name:$end_entity_name})) "
"RETURN p",
            start_entity_name=start_entity_name,
            end_entity_name=end_entity_name
        )
        paths = []
        short_path = 0
        for record in result:
            path = record["p"]
            entities = []
            relations = []
            for i in range(len(path.nodes)):
                node = path.nodes[i]
                entity_name = node["name"]
                entities.append(entity_name)
                if i < len(path.relationships):
                    relationship = path.relationships[i]
                    relation_type = relationship.type
                    relations.append(relation_type)
           
            path_str = ""
            for i in range(len(entities)):
                entities[i] = entities[i].replace("_"," ")
                
                if entities[i] in candidate_list:
                    short_path = 1
                    exist_entity = entities[i]
                path_str += entities[i]
                if i < len(relations):
                    relations[i] = relations[i].replace("_"," ")
                    path_str += "->" + relations[i] + "->"
            
            if short_path == 1:
                paths = [path_str]
                break
            else:
                paths.append(path_str)
                exist_entity = {}
            
        if len(paths) > 5:        
            paths = sorted(paths, key=len)[:5]

        return paths,exist_entity


def combine_lists(*lists):
    combinations = list(itertools.product(*lists))
    results = []
    for combination in combinations:
        new_combination = []
        for sublist in combination:
            if isinstance(sublist, list):
                new_combination += sublist
            else:
                new_combination.append(sublist)
        results.append(new_combination)
    return results


def get_entity_neighbors(entity_name: str,disease_flag) -> List[List[str]]:
    disease = []
    query = """
    MATCH (e:Entity)-[r]->(n)
    WHERE e.name = $entity_name
    RETURN type(r) AS relationship_type,
           collect(n.name) AS neighbor_entities
    """
    result = session.run(query, entity_name=entity_name)

    neighbor_list = []
    for record in result:
        rel_type = record["relationship_type"]
        
        if disease_flag == 1 and rel_type == 'has_symptom':
            continue

        neighbors = record["neighbor_entities"]
        
        if "disease" in rel_type.replace("_"," "):
            disease.extend(neighbors)

        else:
            neighbor_list.append([entity_name.replace("_"," "), rel_type.replace("_"," "), 
                                ','.join([x.replace("_"," ") for x in neighbors])
                                ])
    
    return neighbor_list,disease

def prompt_path_finding(path_input):
    template = """
    There are some knowledge graph path. They follow entity->relationship->entity format.
    \n\n
    {Path}
    \n\n
    Use the knowledge graph information. Try to convert them to natural language, respectively. Use single quotation marks for entity name and relation name. And name them as Path-based Evidence 1, Path-based Evidence 2,...\n\n

    Output:
    """

    prompt = PromptTemplate(
        template = template,
        input_variables = ["Path"]
    )

    system_message_prompt = SystemMessagePromptTemplate(prompt = prompt)
    system_message_prompt.format(Path = path_input)

    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
    chat_prompt_with_values = chat_prompt.format_prompt(Path = path_input,\
                                                        text={})

    # response_of_KG_path = chat(chat_prompt_with_values.to_messages()).content
    response_of_KG_path = chat.invoke(chat_prompt_with_values.to_messages()).content
    return response_of_KG_path

def prompt_neighbor(neighbor):
    template = """
    There are some knowledge graph. They follow entity->relationship->entity list format.
    \n\n
    {neighbor}
    \n\n
    Use the knowledge graph information. Try to convert them to natural language, respectively. Use single quotation marks for entity name and relation name. And name them as Neighbor-based Evidence 1, Neighbor-based Evidence 2,...\n\n

    Output:
    """

    prompt = PromptTemplate(
        template = template,
        input_variables = ["neighbor"]
    )

    system_message_prompt = SystemMessagePromptTemplate(prompt = prompt)
    system_message_prompt.format(neighbor = neighbor)

    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
    chat_prompt_with_values = chat_prompt.format_prompt(neighbor = neighbor,\
                                                        text={})

    response_of_KG_neighbor = chat.invoke(chat_prompt_with_values.to_messages()).content

    return response_of_KG_neighbor

def cosine_similarity_manual(x, y):
    dot_product = np.dot(x, y.T)
    norm_x = np.linalg.norm(x, axis=-1)
    norm_y = np.linalg.norm(y, axis=-1)
    sim = dot_product / (norm_x[:, np.newaxis] * norm_y)
    return sim

def is_unable_to_answer(response):
    try:
        analysis = client.chat.completions.create(
            model="gpt-4.1-2025-04-14",
            messages=[
                {"role": "system", "content": "判断此回答是否无法提供有效信息。直接返回'1'（无法回答）或'0'（可以回答）:"},
                {"role": "user", "content": response}
            ],
            max_tokens=1,
            temperature=0
        )
        score = analysis.choices[0].message.content.strip()
        return score == '1'
    except Exception as e:
        print(f"Error in analysis: {e}")
        return True

def autowrap_text(text, font, max_width):

    text_lines = []
    if font.getsize(text)[0] <= max_width:
        text_lines.append(text)
    else:
        words = text.split(' ')
        i = 0
        while i < len(words):
            line = ''
            while i < len(words) and font.getsize(line + words[i])[0] <= max_width:
                line = line + words[i] + ' '
                i += 1
            if not line:
                line = words[i]
                i += 1
            text_lines.append(line)
    return text_lines

def final_answer(str,response_of_KG_list_path,response_of_KG_neighbor):
    messages  = [
                SystemMessage(content="You are an excellent AI doctor, and you can diagnose diseases and recommend medications based on the symptoms in the conversation. "),
                HumanMessage(content="Patient input:"+ input_text[0]),
                AIMessage(content="You have some medical knowledge information in the following:\n\n" +  '###'+ response_of_KG_list_path + '\n\n' + '###' + response_of_KG_neighbor),
                HumanMessage(content="What disease does the patient have? What tests should patient take to confirm the diagnosis? What recommened medications can cure the disease? Think step by step.\n\n\n"
                            + "Output1: The answer includes disease and tests and recommened medications.\n\n"
                             +"Output2: Show me inference process as a string about extract what knowledge from which Path-based Evidence or Neighor-based Evidence, and in the end infer what result. \n Transport the inference process into the following format:\n Path-based Evidence number('entity name'->'relation name'->...)->Path-based Evidence number('entity name'->'relation name'->...)->Neighbor-based Evidence number('entity name'->'relation name'->...)->Neighbor-based Evidence number('entity name'->'relation name'->...)->result number('entity name')->Path-based Evidence number('entity name'->'relation name'->...)->Neighbor-based Evidence number('entity name'->'relation name'->...). \n\n"
                             +"Output3: Draw a decision tree. The entity or relation in single quotes in the inference process is added as a node with the source of evidence, which is followed by the entity in parentheses.\n\n"
                             + "There is a sample:\n"
                             + """
Output 1:
Based on the symptoms described, the patient may have laryngitis, which is inflammation of the vocal cords. To confirm the diagnosis, the patient should undergo a physical examination of the throat and possibly a laryngoscopy, which is an examination of the vocal cords using a scope. Recommended medications for laryngitis include anti-inflammatory drugs such as ibuprofen, as well as steroids to reduce inflammation. It is also recommended to rest the voice and avoid smoking and irritants.

Output 2:
Path-based Evidence 1('Patient'->'has been experiencing'->'hoarse voice')->Path-based Evidence 2('hoarse voice'->'could be caused by'->'laryngitis')->Neighbor-based Evidence 1('laryngitis'->'requires'->'physical examination of the throat')->Neighbor-based Evidence 2('physical examination of the throat'->'may include'->'laryngoscopy')->result 1('laryngitis')->Path-based Evidence 3('laryngitis'->'can be treated with'->'anti-inflammatory drugs and steroids')->Neighbor-based Evidence 3('anti-inflammatory drugs and steroids'->'should be accompanied by'->'resting the voice and avoiding irritants').

Output 3: 
Patient(Path-based Evidence 1)
└── has been experiencing(Path-based Evidence 1)
    └── hoarse voice(Path-based Evidence 1)(Path-based Evidence 2)
        └── could be caused by(Path-based Evidence 2)
            └── laryngitis(Path-based Evidence 2)(Neighbor-based Evidence 1)
                ├── requires(Neighbor-based Evidence 1)
                │   └── physical examination of the throat(Neighbor-based Evidence 1)(Neighbor-based Evidence 2)
                │       └── may include(Neighbor-based Evidence 2)
                │           └── laryngoscopy(Neighbor-based Evidence 2)(result 1)(Path-based Evidence 3)
                ├── can be treated with(Path-based Evidence 3)
                │   └── anti-inflammatory drugs and steroids(Path-based Evidence 3)(Neighbor-based Evidence 3)
                └── should be accompanied by(Neighbor-based Evidence 3)
                    └── resting the voice and avoiding irritants(Neighbor-based Evidence 3)
                                    """
                             )

                                   ]
        
    # result = chat(messages)
    result = chat.invoke(messages)
    output_all = result.content
    return output_all
def final_answer_nokg(str):
    messages  = [
                SystemMessage(content="You are an excellent AI doctor, and you can diagnose diseases and recommend medications based on the symptoms in the conversation. "),
                HumanMessage(content="Patient input:"+ input_text[0]),
                HumanMessage(content="What disease does the patient have? What tests should patient take to confirm the diagnosis? What recommened medications can cure the disease? Think step by step.\n\n\n"
                            + "Output1: The answer includes disease and tests and recommened medications.\n\n"
                             +"Output2: Show me inference process as a string about extract what knowledge from which Path-based Evidence or Neighor-based Evidence, and in the end infer what result. \n Transport the inference process into the following format:\n Path-based Evidence number('entity name'->'relation name'->...)->Path-based Evidence number('entity name'->'relation name'->...)->Neighbor-based Evidence number('entity name'->'relation name'->...)->Neighbor-based Evidence number('entity name'->'relation name'->...)->result number('entity name')->Path-based Evidence number('entity name'->'relation name'->...)->Neighbor-based Evidence number('entity name'->'relation name'->...). \n\n"
                             +"Output3: Draw a decision tree. The entity or relation in single quotes in the inference process is added as a node with the source of evidence, which is followed by the entity in parentheses.\n\n"
                             + "There is a sample:\n"
                             + """
Output 1:
Based on the symptoms described, the patient may have laryngitis, which is inflammation of the vocal cords. To confirm the diagnosis, the patient should undergo a physical examination of the throat and possibly a laryngoscopy, which is an examination of the vocal cords using a scope. Recommended medications for laryngitis include anti-inflammatory drugs such as ibuprofen, as well as steroids to reduce inflammation. It is also recommended to rest the voice and avoid smoking and irritants.

Output 2:
Path-based Evidence 1('Patient'->'has been experiencing'->'hoarse voice')->Path-based Evidence 2('hoarse voice'->'could be caused by'->'laryngitis')->Neighbor-based Evidence 1('laryngitis'->'requires'->'physical examination of the throat')->Neighbor-based Evidence 2('physical examination of the throat'->'may include'->'laryngoscopy')->result 1('laryngitis')->Path-based Evidence 3('laryngitis'->'can be treated with'->'anti-inflammatory drugs and steroids')->Neighbor-based Evidence 3('anti-inflammatory drugs and steroids'->'should be accompanied by'->'resting the voice and avoiding irritants').

Output 3: 
Patient(Path-based Evidence 1)
└── has been experiencing(Path-based Evidence 1)
    └── hoarse voice(Path-based Evidence 1)(Path-based Evidence 2)
        └── could be caused by(Path-based Evidence 2)
            └── laryngitis(Path-based Evidence 2)(Neighbor-based Evidence 1)
                ├── requires(Neighbor-based Evidence 1)
                │   └── physical examination of the throat(Neighbor-based Evidence 1)(Neighbor-based Evidence 2)
                │       └── may include(Neighbor-based Evidence 2)
                │           └── laryngoscopy(Neighbor-based Evidence 2)(result 1)(Path-based Evidence 3)
                ├── can be treated with(Path-based Evidence 3)
                │   └── anti-inflammatory drugs and steroids(Path-based Evidence 3)(Neighbor-based Evidence 3)
                └── should be accompanied by(Neighbor-based Evidence 3)
                    └── resting the voice and avoiding irritants(Neighbor-based Evidence 3)
                                    """
                             )

                                   ]
        
    # result = chat(messages)
    result = chat.invoke(messages)
    output_all = result.content
    return output_all


def prompt_document(question,instruction):
    template = """
    You are an excellent AI doctor, and you can diagnose diseases and recommend medications based on the symptoms in the conversation.\n\n
    Patient input:\n
    {question}
    \n\n
    You have some medical knowledge information in the following:
    {instruction}
    \n\n
    What disease does the patient have? What tests should patient take to confirm the diagnosis? What recommened medications can cure the disease?
    """

    prompt = PromptTemplate(
        template = template,
        input_variables = ["question","instruction"]
    )

    system_message_prompt = SystemMessagePromptTemplate(prompt = prompt)
    system_message_prompt.format(question = question,
                                 instruction = instruction)

    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
    chat_prompt_with_values = chat_prompt.format_prompt(question = question,\
                                                        instruction = instruction,\
                                                        text={})

    # response_document_bm25 = chat(chat_prompt_with_values.to_messages()).content
    response_document_bm25 = chat.invoke(chat_prompt_with_values.to_messages()).content
    return response_document_bm25

import autogen
import logging
from typing import Dict, List, Optional, Any


# class MedicalQAAutoGenExperiment:
#     """Fixed Medical Question Answering AutoGen Experiment Class"""

#     def __init__(self, openai_api_key: str):
#         self.openai_api_key = openai_api_key
#         self.config_list = [{
#             'model': 'gpt-4.1-2025-04-14',  # Use the latest model for better performance
#             'api_key': openai_api_key
#         }]

#         self.base_llm_config = {
#             "config_list": self.config_list,
#             "cache_seed": None,
#             "temperature": 0.2,
#             "top_p": 0.9,
#             "max_tokens": 1500,
#             "timeout": 180,
#         }

#     def run_multi_agent_experiment(self, user_message: str) -> str:
#         """Fixed multi-agent experiment with proper collaboration"""
#         try:
#             specialist_config = self.base_llm_config.copy()
#             specialist_config["cache_seed"] = 100
#             specialist_config["temperature"] = 0.15

#             reviewer_config = self.base_llm_config.copy()
#             reviewer_config["cache_seed"] = 200
#             reviewer_config["temperature"] = 0.25

#             coordinator_config = self.base_llm_config.copy()
#             coordinator_config["cache_seed"] = 300
#             coordinator_config["temperature"] = 0.1

#             medical_specialist = autogen.AssistantAgent(
#                 name="Medical_Specialist",
#                 llm_config=specialist_config,
#                 system_message="""You are a Senior Medical Diagnostician specializing in systematic clinical analysis.

# CORE RESPONSIBILITIES:
# - Analyze symptoms using structured differential diagnosis approach
# - Prioritize diagnoses by likelihood and clinical severity
# - Recommend evidence-based diagnostic workup
# - Suggest appropriate treatment protocols
# - Consider contraindications and drug interactions

# RESPONSE FORMAT:
# 1. CHIEF COMPLAINT ANALYSIS: [Brief symptom summary]
# 2. DIFFERENTIAL DIAGNOSIS: [List 3-5 conditions ranked by probability]
# 3. RECOMMENDED TESTS: [Specific diagnostic procedures with rationale]
# 4. TREATMENT PLAN: [Medications, procedures, lifestyle modifications]
# 5. FOLLOW-UP: [Monitoring requirements and red flags]

# Be thorough but concise. Focus on actionable clinical recommendations. After your analysis, state "Awaiting review from Clinical_Reviewer."
# """
#             )

#             clinical_reviewer = autogen.AssistantAgent(
#                 name="Clinical_Reviewer",
#                 llm_config=reviewer_config,
#                 system_message="""You are a Clinical Quality Assurance Specialist focused on safety and accuracy. Review the Medical_Specialist's analysis.

# REVIEW CRITERIA:
# - Diagnostic reasoning completeness and accuracy
# - Treatment safety and appropriateness
# - Missing considerations or alternative diagnoses
# - Risk stratification adequacy
# - Guideline compliance

# RESPONSE REQUIREMENTS:
# If the specialist's analysis is comprehensive and safe, respond: "APPROVED: Analysis meets clinical standards. Ready for Medical_Coordinator synthesis."

# If improvements are needed, provide:
# 1. GAPS IDENTIFIED: [Specific missing elements]
# 2. SAFETY CONCERNS: [Potential risks or contraindications]
# 3. ADDITIONAL CONSIDERATIONS: [Alternative diagnoses or treatments]
# 4. RECOMMENDATIONS FOR Medical_Specialist: [Specific improvements needed for the Medical_Specialist to address]
# And conclude with "Awaiting revised analysis from Medical_Specialist."

# Only approve when confident in the clinical safety and completeness.
# """
#             )

#             medical_coordinator = autogen.AssistantAgent(
#                 name="Medical_Coordinator",
#                 llm_config=coordinator_config,
#                 system_message="""You are a Medical Case Coordinator. Your role is to synthesize the Medical_Specialist's analysis *after* it has been APPROVED by the Clinical_Reviewer. Do not act before this approval.

# COORDINATION TASKS:
# - Integrate specialist diagnosis with reviewer feedback (if any implied by approval).
# - Ensure comprehensive patient care plan.
# - Prioritize urgent vs routine interventions.
# - Provide clear patient communication summary.

# FINAL RESPONSE FORMAT:
# ## CLINICAL ASSESSMENT
# [Integrated diagnosis and reasoning based on specialist's approved report]

# ## IMMEDIATE ACTIONS
# [Urgent tests/treatments needed based on specialist's approved report]

# ## TREATMENT PLAN
# [Comprehensive management strategy based on specialist's approved report]

# ## PATIENT GUIDANCE
# [Clear instructions for patient based on specialist's approved report]

# Wait for the Clinical_Reviewer to state "APPROVED: Analysis meets clinical standards. Ready for Medical_Coordinator synthesis." before you provide your synthesis. Your response should be the final message in this consultation.
# """
#             )

#             case_manager = autogen.UserProxyAgent(
#                 name="Case_Manager",
#                 human_input_mode="NEVER",
#                 max_consecutive_auto_reply=0, # Correct: Case_Manager only initiates.
#                 code_execution_config=False,
#                 system_message="You are a Case Manager. You will initiate the medical case discussion with the patient's query. Then, the Medical_Specialist, Clinical_Reviewer, and Medical_Coordinator will proceed. You will not intervene further."
#             )

#             # Termination condition focused on Medical_Coordinator's final output
#             def is_termination_msg(x: Dict[str, Any]) -> bool:
#                 content = x.get("content", "").strip()
#                 # Terminate if the message is from Medical_Coordinator AND contains its specific headers.
#                 # This makes termination much more specific and tied to the desired final output.
#                 return x.get("name") == "Medical_Coordinator" and \
#                        ("## CLINICAL ASSESSMENT" in content and "## PATIENT GUIDANCE" in content)

#             group_chat = autogen.GroupChat(
#                 agents=[case_manager, medical_specialist, clinical_reviewer, medical_coordinator],
#                 messages=[],
#                 max_round=12, # Increased for potentially more back-and-forth during review
#                 speaker_selection_method="auto",
#                 allow_repeat_speaker=True, # Necessary if reviewer asks for revisions
#                 # Removed: group_chat.is_termination_msg = is_termination_msg.
#                 # Instead, pass it to GroupChatManager or rely on agent prompts for termination keywords if needed.
#                 # For now, we will use is_termination_msg in the GroupChatManager
#             )

#             manager = autogen.GroupChatManager(
#                 groupchat=group_chat,
#                 llm_config=self.base_llm_config, # Manager can use a general config
#                 is_termination_msg=is_termination_msg, # Pass termination logic here
#                 system_message="""You are the GroupChat Manager. Facilitate a structured medical consultation ensuring the following flow:
# 1.  The Case_Manager will provide the initial user_message.
# 2.  Medical_Specialist provides initial analysis and then states "Awaiting review from Clinical_Reviewer."
# 3.  Clinical_Reviewer evaluates.
#     a. If approved, Clinical_Reviewer states "APPROVED: Analysis meets clinical standards. Ready for Medical_Coordinator synthesis."
#     b. If revisions needed, Clinical_Reviewer states "Awaiting revised analysis from Medical_Specialist." and Medical_Specialist should revise. This loop (Specialist revision -> Reviewer check) can continue.
# 4.  Once Clinical_Reviewer approves, Medical_Coordinator synthesizes the final recommendations using the specified "FINAL RESPONSE FORMAT". The conversation ends after the Medical_Coordinator's synthesis.
# Ensure agents stick to their roles and the prescribed conversational flow. Only the Medical_Coordinator's final formatted response should trigger termination.
# """
#             )

#             chat_result = case_manager.initiate_chat(
#                 manager,
#                 message=user_message,
#                 # silent=False # Set to True for production, False for debugging to see all messages
#             )

#             # Extract the final coordinated response
#             # The termination condition should ensure the last message is from the coordinator.
#             if chat_result.chat_history:
#                 last_message = chat_result.chat_history[-1]
#                 if last_message.get('name') == 'Medical_Coordinator' and \
#                    ("## CLINICAL ASSESSMENT" in last_message.get('content', '') and \
#                     "## PATIENT GUIDANCE" in last_message.get('content', '')):
#                     return last_message.get('content', '')
#                 else:
#                     # If the last message isn't the coordinator's structured output,
#                     # the process likely didn't complete as expected.
#                     logger.warning("Multi-agent chat did not end with Medical_Coordinator's final synthesis.")
#                     # You might want to return the whole chat history for debugging or a more specific error.
#                     # For now, let's find the last message from the coordinator if any, otherwise the very last message.
#                     for msg in reversed(chat_result.chat_history):
#                         if msg.get('name') == 'Medical_Coordinator':
#                             return msg.get('content', "Error: Coordinator spoke but content missing.")
#                     return chat_result.summary if chat_result.summary else "Multi-agent collaboration did not produce a final coordinated response as expected."

#             return "Multi-agent collaboration did not produce a final response or chat history was empty."

#         except Exception as e:
#             logger.error(f"Multi-agent execution failed: {e}", exc_info=True)
#             return f"Multi-agent execution failed: {str(e)}"

#     def run_single_agent_experiment(self, user_message: str) -> str:
#         """Enhanced single agent with comparable sophistication"""
#         try:
#             enhanced_config = self.base_llm_config.copy()
#             enhanced_config["cache_seed"] = 42
#             enhanced_config["temperature"] = 0.15
#             enhanced_config["max_tokens"] = 2000

#             assistant = autogen.AssistantAgent(
#                 name="AI_Medical_Expert",
#                 llm_config=enhanced_config,
#                 system_message="""You are an expert AI Medical Consultant with comprehensive clinical expertise.

# ANALYTICAL FRAMEWORK:
# 1. SYMPTOM ANALYSIS: Systematically evaluate presenting complaints
# 2. DIFFERENTIAL DIAGNOSIS: Consider multiple possibilities with probability ranking
# 3. DIAGNOSTIC WORKUP: Recommend appropriate tests and examinations
# 4. TREATMENT PLANNING: Evidence-based therapeutic recommendations
# 5. RISK ASSESSMENT: Safety considerations and monitoring requirements

# RESPONSE STRUCTURE:
# ## CLINICAL ANALYSIS
# [Detailed symptom evaluation and pattern recognition]

# ## DIFFERENTIAL DIAGNOSIS
# [Ranked list of possible conditions with reasoning]

# ## DIAGNOSTIC RECOMMENDATIONS
# [Specific tests needed with clinical rationale]

# ## TREATMENT PLAN
# [Medications, procedures, lifestyle modifications]

# ## FOLLOW-UP CARE
# [Monitoring schedule and warning signs]

# Provide comprehensive, evidence-based medical guidance prioritizing patient safety."""
#             )

#             user_proxy = autogen.UserProxyAgent(
#                 name="Medical_Query_Handler", # Changed name for clarity
#                 human_input_mode="NEVER",
#                 max_consecutive_auto_reply=1, # For single assistant, user proxy replies once to get response
#                 code_execution_config=False,
#             )

#             chat_result = user_proxy.initiate_chat(
#                 assistant,
#                 message=user_message,
#                 # silent=False
#             )

#             if chat_result.chat_history:
#                 # The assistant's response should be the last message in this simple 2-turn exchange
#                 return chat_result.chat_history[-1].get('content', 'No response generated by AI_Medical_Expert')
#             elif hasattr(chat_result, 'summary') and chat_result.summary: # Fallback
#                 return chat_result.summary

#             return "Single agent did not produce a valid response"

#         except Exception as e:
#             logger.error(f"Single agent execution failed: {e}", exc_info=True)
#             return f"Single agent execution failed: {str(e)}"

#     def create_medical_prompt_template(self, patient_question: str, path_evidence: str, neighbor_evidence: str) -> str:
#         prompt = f"""PATIENT SYMPTOM REPORT:
# {patient_question}

# PATH-BASED EVIDENCE:
# {path_evidence}

# NEIGHBOR-BASED EVIDENCE:
# {neighbor_evidence}

# Please analyze the patient's case based on both the reported symptoms and the structured medical evidence provided above. This information should be used by the Medical_Specialist for their initial assessment."""
#         return prompt

# Keep all your existing imports and helper functions
# ... (from dotenv import load_dotenv ... up to the MedicalQAAutoGenExperiment class)
import logging # Ensure logger is imported for the class
logger = logging.getLogger(__name__) # Standard way to get a logger

class MedicalQAAutoGenExperiment:
    """Fixed Medical Question Answering AutoGen Experiment Class"""

    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        # Ensure you are using the model intended for these experiments.
        # Your chat_4o_mini uses "gpt-4.1-2025-04-14", so it's consistent here.
        self.config_list = [{
            'model': 'gpt-4.1-2025-04-14',
            'api_key': openai_api_key
        }]

        self.base_llm_config = {
            "config_list": self.config_list,
            "cache_seed": None, # Consider setting a seed for reproducibility during debugging
            "temperature": 0.1, # Lowered temperature for more deterministic medical output
            "top_p": 0.9,       # Can keep top_p as is or slightly lower
            "max_tokens": 2000, # Increased for potentially more detailed multi-agent responses
            "timeout": 180,
        }

    def run_multi_agent_experiment(self, user_message: str) -> str:
        """Fixed multi-agent experiment with improved information handling and collaboration"""
        try:
            # Configurations for each agent - can be tuned further
            specialist_config = self.base_llm_config.copy()
            specialist_config["cache_seed"] = 101 # Different seeds can sometimes help
            specialist_config["temperature"] = 0.05 # Specialist needs to be very factual

            reviewer_config = self.base_llm_config.copy()
            reviewer_config["cache_seed"] = 202
            reviewer_config["temperature"] = 0.1

            coordinator_config = self.base_llm_config.copy()
            coordinator_config["cache_seed"] = 303
            coordinator_config["temperature"] = 0.05

            # --- Enhanced Agent System Messages ---

            medical_specialist = autogen.AssistantAgent(
                name="Medical_Specialist",
                llm_config=specialist_config,
                system_message="""You are a Senior Medical Diagnostician. Your primary task is to analyze the patient's case based on the information provided in the user message.
The user message will contain three key sections:
1.  **PATIENT SYMPTOM REPORT**: The patient's own description of their symptoms.
2.  **PATH-BASED EVIDENCE**: Structured information from a medical knowledge graph detailing relationships between entities.
3.  **NEIGHBOR-BASED EVIDENCE**: Structured information from a medical knowledge graph detailing entities connected to key concepts.

YOUR CORE RESPONSIBILITIES:
-   Thoroughly analyze the **PATIENT SYMPTOM REPORT**.
-   Critically **integrate the PATH-BASED EVIDENCE and NEIGHBOR-BASED EVIDENCE** into your assessment. You MUST explicitly state how this KG evidence supports, refutes, or refines potential diagnoses or lines of inquiry. Do not ignore this evidence.
-   Formulate a differential diagnosis, prioritizing conditions based on the synthesis of ALL available information (symptoms + KG evidence).
-   Recommend appropriate diagnostic tests based on your differential diagnosis and the supporting evidence.
-   Suggest a preliminary treatment approach, considering the evidence.

MANDATORY OUTPUT STRUCTURE:
1.  **PATIENT CASE OVERVIEW**:
    * Briefly summarize the patient's reported symptoms.
    * Highlight the most salient points from PATH-BASED and NEIGHBOR-BASED evidence that you will use in your analysis.
2.  **INTEGRATED CLINICAL ANALYSIS & DIFFERENTIAL DIAGNOSIS**:
    * Discuss how the patient's symptoms correlate with the provided PATH-BASED and NEIGHBOR-BASED KG evidence.
    * Provide a ranked list of 3-4 most likely differential diagnoses. For each diagnosis, explain your reasoning, explicitly detailing how BOTH symptoms AND specific KG evidence (cite which parts, e.g., "Path-Evidence item X suggests Y", "Neighbor-Evidence Z points to W") support this possibility.
3.  **RECOMMENDED DIAGNOSTIC WORKUP**:
    * List specific diagnostic tests required to confirm or rule out your differential diagnoses.
    * Provide a brief rationale for each test, linking it back to the differential diagnoses and, if applicable, the KG evidence.
4.  **PRELIMINARY TREATMENT & MANAGEMENT CONSIDERATIONS**:
    * Suggest potential treatment avenues or management strategies based on the likely diagnoses and available evidence.
5.  **SELF-ASSESSMENT OF KG EVIDENCE UTILIZATION**:
    * Briefly state how confident you are that you have effectively and accurately incorporated the PATH-BASED and NEIGHBOR-BASED evidence into your analysis (e.g., "High confidence, KG evidence was pivotal for diagnosis X"; "Moderate confidence, KG evidence provided supporting details but was not primary driver for diagnosis Y").

After completing your analysis strictly following this structure, conclude with the EXACT phrase: "Awaiting review from Clinical_Reviewer."
"""
            )

            clinical_reviewer = autogen.AssistantAgent(
                name="Clinical_Reviewer",
                llm_config=reviewer_config,
                system_message="""You are a Clinical Quality Assurance Specialist. Your role is to meticulously review the Medical_Specialist's analysis. The Medical_Specialist was instructed to integrate patient symptoms with provided PATH-BASED and NEIGHBOR-BASED Knowledge Graph (KG) evidence.

YOUR REVIEW CRITERIA:
-   **Evidence Integration**: CRITICALLY ASSESS if the Medical_Specialist effectively and accurately integrated the PATH-BASED and NEIGHBOR-BASED KG evidence with the patient's symptoms. Was the KG evidence explicitly referenced and used in forming differential diagnoses? Was its significance correctly interpreted?
-   **Diagnostic Reasoning**: Evaluate the completeness, accuracy, and logical flow of the specialist's diagnostic reasoning. Are the differential diagnoses plausible given ALL evidence?
-   **Treatment Safety & Appropriateness**: Assess the safety and suitability of the recommended tests and preliminary treatment considerations.
-   **Completeness**: Are there any glaring omissions, missing considerations, or alternative diagnoses that should have been explored, especially in light of the KG evidence?
-   **Guideline Compliance**: Does the analysis align with general evidence-based medical principles?

RESPONSE REQUIREMENTS:
If the specialist's analysis is comprehensive, accurately integrates all evidence (especially KG data), and is clinically sound:
Respond with: "APPROVED: Analysis demonstrates strong evidence integration and meets clinical standards. Ready for Medical_Coordinator synthesis."

If improvements are URGENTLY needed (e.g., misinterpretation of critical KG data, unsafe suggestions):
1.  **CRITICAL GAPS/ERRORS**: [Clearly state any critical issues, especially concerning KG evidence misuse or patient safety.]
2.  **RECOMMENDATIONS FOR REVISION**: [Provide specific, actionable instructions for the Medical_Specialist to address these critical issues.]
And conclude with: "NEEDS URGENT REVISION by Medical_Specialist."

If non-critical improvements are suggested:
1.  **GAPS IDENTIFIED**: [Specific missing elements or areas for minor refinement, e.g., "Further clarification on how KG path X relates to symptom Y would be beneficial."]
2.  **SUGGESTIONS FOR ENHANCEMENT**: [Constructive feedback for the Medical_Specialist.]
And conclude with: "Awaiting revised analysis from Medical_Specialist."

Focus on ensuring the KG evidence was meaningfully used. Only approve analyses that demonstrate this.
"""
            )

            medical_coordinator = autogen.AssistantAgent(
                name="Medical_Coordinator",
                llm_config=coordinator_config,
                system_message="""You are a Medical Case Coordinator. Your role is to synthesize the Medical_Specialist's analysis ONLY AFTER it has been explicitly APPROVED by the Clinical_Reviewer. Do not act before receiving the "APPROVED" statement. The Specialist's report is expected to integrate patient symptoms with Knowledge Graph (KG) evidence.

COORDINATION TASKS (upon receiving "APPROVED" status):
-   Carefully review the approved analysis from the Medical_Specialist.
-   Synthesize this information into a final, clear, patient-facing summary.
-   Ensure the summary reflects the integration of symptoms and KG evidence as highlighted in the specialist's approved report.
-   Prioritize urgent vs. routine interventions if specified.

FINAL RESPONSE FORMAT (to be used for the patient):
## FINAL CLINICAL ASSESSMENT
[Synthesize the primary diagnosis and key reasoning from the specialist's approved report. Clearly mention how KG evidence contributed if detailed in the report.]

## RECOMMENDED ACTIONS & DIAGNOSTICS
[Summarize the recommended tests and immediate actions from the specialist's approved report.]

## PRELIMINARY TREATMENT & MANAGEMENT PLAN
[Outline the comprehensive management strategy from the specialist's approved report.]

## PATIENT GUIDANCE & NEXT STEPS
[Provide clear instructions and advice for the patient based on the approved findings.]

Wait for the Clinical_Reviewer to state "APPROVED: Analysis demonstrates strong evidence integration and meets clinical standards. Ready for Medical_Coordinator synthesis." before you provide your synthesis. Your response should be the final message in this consultation.
"""
            )

            case_manager = autogen.UserProxyAgent(
                name="Case_Manager",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=0,
                code_execution_config=False,
                system_message="You are a Case Manager. You will initiate the medical case discussion by providing the patient's query, which includes their symptom report and relevant Knowledge Graph evidence. After initiation, you will not intervene further. The specialist team will handle the case."
            )

            def is_termination_msg(x: Dict[str, Any]) -> bool:
                content = x.get("content", "").strip()
                is_coordinator = x.get("name") == "Medical_Coordinator"
                # Check for key headers that the Medical_Coordinator is supposed to output
                has_assessment_header = "## FINAL CLINICAL ASSESSMENT" in content
                has_guidance_header = "## PATIENT GUIDANCE & NEXT STEPS" in content
                return is_coordinator and has_assessment_header and has_guidance_header

            group_chat = autogen.GroupChat(
                agents=[case_manager, medical_specialist, clinical_reviewer, medical_coordinator],
                messages=[],
                max_round=10, # Reduced slightly, as more focused prompts might lead to quicker resolution
                speaker_selection_method="auto", # "auto" should work with the keyword-based handoffs
                allow_repeat_speaker=True, # Important for reviewer-specialist interaction
            )

            # System message for the manager to enforce the flow
            manager_system_message = """You are the GroupChat Manager. Your role is to strictly facilitate the medical consultation ensuring the following flow:
1.  The Case_Manager provides the initial user_message (patient query + KG evidence).
2.  Medical_Specialist analyzes this input, deeply integrating KG evidence, and concludes with "Awaiting review from Clinical_Reviewer."
3.  Clinical_Reviewer evaluates the specialist's analysis.
    a.  If approved, Clinical_Reviewer states "APPROVED: Analysis demonstrates strong evidence integration and meets clinical standards. Ready for Medical_Coordinator synthesis."
    b.  If revisions are needed (critical or minor), Clinical_Reviewer states the appropriate message (e.g., "NEEDS URGENT REVISION by Medical_Specialist." or "Awaiting revised analysis from Medical_Specialist."). The Medical_Specialist must then revise. This loop can continue.
4.  Once Clinical_Reviewer approves, Medical_Coordinator synthesizes the final recommendations using its specified "FINAL RESPONSE FORMAT".
The conversation terminates ONLY after the Medical_Coordinator provides its complete, formatted response. Ensure agents adhere to their roles and use the exact trigger phrases for handoffs.
The Medical_Specialist MUST address the initial user_message containing PATIENT SYMPTOM REPORT, PATH-BASED EVIDENCE, and NEIGHBOR-BASED EVIDENCE.
The Clinical_Reviewer MUST verify the Medical_Specialist's integration of this evidence.
"""
            manager = autogen.GroupChatManager(
                groupchat=group_chat,
                llm_config=self.base_llm_config,
                is_termination_msg=is_termination_msg,
                system_message=manager_system_message
            )

            chat_result = case_manager.initiate_chat(
                manager,
                message=user_message, # This is the common_user_input from your main script
                # silent=True # Set to False for debugging to see all messages
            )

            if chat_result.chat_history:
                last_message = chat_result.chat_history[-1]
                # Check if the last message is indeed the structured output from Medical_Coordinator
                if last_message.get('name') == 'Medical_Coordinator' and \
                   ("## FINAL CLINICAL ASSESSMENT" in last_message.get('content', '') and \
                    "## PATIENT GUIDANCE & NEXT STEPS" in last_message.get('content', '')):
                    return last_message.get('content', "Error: Coordinator spoke but content missing from structured output.")
                else:
                    # If termination happened for other reasons (e.g., max_round) or if the Coordinator didn't output correctly
                    logger.warning("Multi-agent chat did not end with Medical_Coordinator's complete final synthesis. Checking for any Coordinator message.")
                    for msg in reversed(chat_result.chat_history):
                        if msg.get('name') == 'Medical_Coordinator' and msg.get('content'):
                            logger.warning(f"Returning last message from Medical_Coordinator: {msg.get('content')[:100]}...")
                            return msg.get('content')
                    # Fallback to summary or a more generic error
                    final_content = chat_result.summary if chat_result.summary else "Multi-agent collaboration did not produce the expected final coordinated response."
                    logger.warning(f"Returning fallback response: {final_content[:100]}...")
                    return final_content
            return "Multi-agent collaboration did not produce a final response or chat history was empty."

        except Exception as e:
            logger.error(f"Multi-agent execution failed: {e}", exc_info=True)
            return f"Multi-agent execution failed: {str(e)}"

    def run_single_agent_experiment(self, user_message: str) -> str:
        """Enhanced single agent with comparable sophistication"""
        try:
            enhanced_config = self.base_llm_config.copy()
            enhanced_config["cache_seed"] = 42
            enhanced_config["temperature"] = 0.05 # Make it more factual like specialist
            enhanced_config["max_tokens"] = 2500 # Allow more space for comprehensive answer

            # The single agent's system message already expects a structured input if present in user_message
            assistant = autogen.AssistantAgent(
                name="AI_Medical_Expert",
                llm_config=enhanced_config,
                system_message="""You are an expert AI Medical Consultant with comprehensive clinical expertise.
You will receive a user message that may include:
1.  **PATIENT SYMPTOM REPORT**
2.  **PATH-BASED EVIDENCE** (from a knowledge graph)
3.  **NEIGHBOR-BASED EVIDENCE** (from a knowledge graph)

ANALYTICAL FRAMEWORK & RESPONSE STRUCTURE:
If PATH-BASED or NEIGHBOR-BASED evidence is provided, you MUST integrate it into your analysis and explicitly state how it influences your conclusions.

## 1. CLINICAL ANALYSIS & EVIDENCE INTEGRATION
    * Evaluate the presenting complaints from the PATIENT SYMPTOM REPORT.
    * Detail how the PATH-BASED EVIDENCE and NEIGHBOR-BASED EVIDENCE (if provided) support, refute, or refine your understanding of the case. Be specific.

## 2. DIFFERENTIAL DIAGNOSIS
    * Provide a ranked list of possible conditions. For each, explain your reasoning, clearly linking symptoms and any provided KG evidence.

## 3. DIAGNOSTIC RECOMMENDATIONS
    * List specific tests or examinations needed. Justify each recommendation based on your differential diagnosis and the supporting evidence (including KG).

## 4. TREATMENT PLAN
    * Suggest evidence-based therapeutic options, lifestyle modifications, or procedures.

## 5. FOLLOW-UP CARE & RISK ASSESSMENT
    * Outline monitoring requirements, warning signs, and any safety considerations.

Provide comprehensive, evidence-based medical guidance, prioritizing patient safety and the explicit use of all provided information.
"""
            )

            user_proxy = autogen.UserProxyAgent(
                name="Medical_Query_Handler",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=1, # User proxy sends message, assistant replies, then stops.
                code_execution_config=False,
            )

            chat_result = user_proxy.initiate_chat(
                assistant,
                message=user_message, # This is common_user_input
                # silent=True
            )

            if chat_result.chat_history and len(chat_result.chat_history) > 1:
                # The assistant's response should be the last message.
                # initiate_chat adds the initial message, so history[-1] is the assistant's reply.
                assistant_response = chat_result.chat_history[-1].get('content')
                if assistant_response:
                    return assistant_response
                else:
                    logger.warning("AI_Medical_Expert responded with empty content.")
                    return "AI_Medical_Expert provided an empty response."
            elif hasattr(chat_result, 'summary') and chat_result.summary:
                logger.warning("Single agent chat history was short or problematic, returning summary.")
                return chat_result.summary
            else:
                logger.warning("Single agent did not produce a valid response or chat history.")
                return "Single agent did not produce a valid response."

        except Exception as e:
            logger.error(f"Single agent execution failed: {e}", exc_info=True)
            return f"Single agent execution failed: {str(e)}"

    def create_medical_prompt_template(self, patient_question: str, path_evidence: str, neighbor_evidence: str) -> str:
        # This template is good. It clearly structures the input for the agents.
        # The crucial part is ensuring the agents are prompted to USE these sections.
        prompt = f"""Please analyze the following patient case:

**PATIENT SYMPTOM REPORT:**
{patient_question}

**PATH-BASED EVIDENCE (from Knowledge Graph):**
{path_evidence if path_evidence else "No specific path-based evidence provided."}

**NEIGHBOR-BASED EVIDENCE (from Knowledge Graph):**
{neighbor_evidence if neighbor_evidence else "No specific neighbor-based evidence provided."}

This information (symptoms and KG evidence) should be used by the Medical_Specialist for their initial assessment, or by the AI_Medical_Expert directly.
The analysis should thoroughly integrate all three sections.
"""
        return prompt

# ================== Self-Consistency 实现 ==================
# Optimized Self-Consistency implementation - add this to your code before the main section

def generate_with_self_consistency(input_text, response_of_KG_list_path, response_of_KG_neighbor, num_samples=5):
    """Generate diagnosis through self-consistency approach with multiple samples and voting"""
    print(f"Generating {num_samples} samples for self-consistency...")
    
    candidates = []
    answers = []
    
    for i in range(num_samples):
        try:
            print(f"Generating sample {i+1}/{num_samples}...")
            # Use a higher temperature for diversity in responses
            messages = [
                {"role": "system", "content": "You are an excellent AI doctor, and you can diagnose diseases and recommend medications based on the symptoms in the conversation."},
                {"role": "user", "content": "Patient input:"+ input_text},
                {"role": "assistant", "content": "You have some medical knowledge information in the following:\n\n" +  
                                                '###'+ response_of_KG_list_path + '\n\n' + '###' + response_of_KG_neighbor},
                {"role": "user", "content": "What disease does the patient have? What tests should patient take to confirm the diagnosis? What recommened medications can cure the disease? Think step by step."}
            ]
            
            response = client.chat.completions.create(
                model="gpt-4.1-2025-04-14",
                messages=messages,
                temperature=0.5  # Higher temperature for diversity
            )
            
            full_response = response.choices[0].message.content
            candidates.append(full_response)
            
            # Extract the core diagnosis answer
            core_answer = extract_final_answer(full_response)
            answers.append(core_answer)
            
        except Exception as e:
            print(f"Error generating sample {i+1}: {e}")
            # Continue with other samples
    
    # If we couldn't generate any samples, return a fallback
    if not candidates:
        return "Unable to generate diagnosis due to API limitations.", []
    
    # Simple majority voting for consensus
    # First, extract key diagnoses using a more robust method
    diagnoses = extract_key_diagnoses(answers)
    
    if diagnoses:
        # Find most common diagnosis
        from collections import Counter
        most_common = Counter(diagnoses).most_common(1)[0][0]
        
        # Find the full response that contains this diagnosis
        for i, answer in enumerate(answers):
            if most_common.lower() in answer.lower():
                consensus = candidates[i]
                break
        else:
            consensus = candidates[0]  # Fallback to first answer
    else:
        # If no clear diagnoses extracted, use the first response
        consensus = candidates[0]
        
    return consensus, candidates

def extract_final_answer(response):
    """Extract the core diagnosis part from a response"""
    # Try to find structured output section
    match = re.search(r"Output 1:(.*?)(Output 2:|$)", response, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    
    # If no structured output, return the whole response
    return response

def extract_key_diagnoses(answers):
    """Extract the key disease diagnoses from a list of answers"""
    diagnoses = []
    
    # Common patterns to identify disease mentions
    patterns = [
        r"(?:diagnosis|disease|condition|suffering from|patient has|diagnosed with)\s+(?:is|of|with)?\s+([A-Za-z\s]+)",
        r"(?:likely|probably|possibly|definitely|certainly|suspected)\s+([A-Za-z\s]+)",
        r"patient may have\s+([A-Za-z\s]+)"
    ]
    
    for answer in answers:
        for pattern in patterns:
            matches = re.findall(pattern, answer, re.IGNORECASE)
            for match in matches:
                # Clean up the extracted disease name
                disease = match.strip().rstrip('.,:;')
                if len(disease) > 3 and len(disease) < 50:  # Reasonable length for disease name
                    diagnoses.append(disease)
    
    return diagnoses


# import re
# import time
# import random
# from typing import List, Dict, Any, Callable, Tuple, Optional
# import numpy as np
# ##################--------TOT---------------####################
# # ================== Thought Generator Methods ==================

# def thought_generator_sampling(input_text, medical_knowledge, num_samples=3, focus_area=None):
#     """Enhanced medical thought sampling generator with better structure"""
#     focus_prompt = f"Pay special attention to {focus_area} aspects." if focus_area else ""
    
#     messages = [
#         {"role": "system", "content": f"""You are a professional medical AI assistant. Generate diverse, high-quality diagnostic reasoning paths.
# Each path should follow structured medical reasoning: Chief Complaint → History → Physical Exam → Differential Diagnosis → Tests → Final Diagnosis.
# {focus_prompt} Ensure each path explores different possibilities and reasoning approaches.
# Be specific about medical conditions, diagnostic criteria, and clinical decision-making."""},
        
#         {"role": "user", "content": f"""Patient Case: {input_text}

# Available Medical Knowledge: {medical_knowledge}

# Generate {num_samples} distinct diagnostic reasoning paths. Each path should:
# 1. Start with "DIAGNOSTIC PATH #X:"
# 2. Follow systematic medical reasoning
# 3. Consider different differential diagnoses with specific medical terminology
# 4. Suggest appropriate diagnostic tests with rationale
# 5. Provide confidence level and clinical reasoning for conclusions
# 6. Reference relevant medical knowledge provided

# Format each path clearly and ensure they explore different diagnostic possibilities."""}
#     ]
    
#     try:
#         response = client.chat.completions.create(
#             model="gpt-4.1-2025-04-14",
#             messages=messages,
#             temperature=0.7,
#             max_tokens=2000
#         )
#         content = response.choices[0].message.content
        
#         # Enhanced path extraction with multiple fallback strategies
#         paths = re.findall(r"DIAGNOSTIC PATH #\d+:.*?(?=DIAGNOSTIC PATH #\d+:|$)", content, re.DOTALL)
#         cleaned_paths = [path.strip() for path in paths if path.strip() and len(path.strip()) > 100]
        
#         # Fallback strategies for better content extraction
#         if len(cleaned_paths) < num_samples:
#             # Try alternative markers
#             alt_paths = re.findall(r"(?:Path|Approach) \d+:.*?(?=(?:Path|Approach) \d+:|$)", content, re.DOTALL)
#             cleaned_paths.extend([p.strip() for p in alt_paths if p.strip() and len(p.strip()) > 100])
        
#         if len(cleaned_paths) < num_samples:
#             # Split by double newlines and filter meaningful content
#             paragraphs = content.split('\n\n')
#             meaningful_paras = [p.strip() for p in paragraphs if len(p.strip()) > 150 and any(term in p.lower() for term in ['diagnosis', 'symptom', 'treatment', 'test', 'condition'])]
#             cleaned_paths.extend(meaningful_paras)
        
#         # Ensure quality over quantity - add meaningful fallbacks only if needed
#         while len(cleaned_paths) < num_samples:
#             fallback_content = f"""Alternative diagnostic approach #{len(cleaned_paths)+1}:
# Based on symptoms: {input_text}
# Consider differential diagnoses including common and rare conditions.
# Systematic evaluation should include detailed history, physical examination, and appropriate diagnostic testing.
# Clinical reasoning should integrate available medical knowledge: {medical_knowledge[:200]}..."""
#             cleaned_paths.append(fallback_content)
            
#         return cleaned_paths[:num_samples]
        
#     except Exception as e:
#         print(f"Error in thought generation: {e}")
#         # Provide meaningful fallback responses
#         fallback_paths = []
#         for i in range(num_samples):
#             fallback_paths.append(f"""DIAGNOSTIC PATH #{i+1}:
# Clinical presentation: {input_text}
# Systematic approach required for proper diagnosis.
# Consider multiple differential diagnoses based on presenting symptoms.
# Utilize available medical knowledge for evidence-based decision making.
# Recommend appropriate diagnostic tests and follow clinical protocols.""")
#         return fallback_paths

# def thought_generator_sequential(input_text, medical_knowledge, num_samples=3):
#     """Enhanced sequential medical thought generator with better diversity and clinical focus"""
#     thoughts = []
#     explored_concepts = set()
    
#     base_context = f"Patient case: {input_text}\nMedical knowledge: {medical_knowledge}"
    
#     # Define clinical focus areas for sequential exploration
#     clinical_focus_areas = [
#         "primary_differential_diagnosis",
#         "secondary_conditions_comorbidities", 
#         "diagnostic_testing_strategy",
#         "treatment_planning_approach"
#     ]
    
#     for i in range(num_samples):
#         current_focus = clinical_focus_areas[i % len(clinical_focus_areas)]
        
#         # Build exclusion context more intelligently
#         exclusion_prompt = ""
#         if explored_concepts:
#             exclusion_prompt = f"\nExplore different aspects from: {', '.join(list(explored_concepts)[:5])}"
        
#         previous_context = ""
#         if thoughts:
#             # Summarize previous approaches rather than full text
#             previous_summaries = []
#             for j, t in enumerate(thoughts):
#                 summary = t[:] 
#                 previous_summaries.append(f"Approach {j+1}: {summary}")
#             previous_context = f"\nPreviously explored:\n" + "\n".join(previous_summaries)
        
#         messages = [
#             {"role": "system", "content": f"""You are a senior medical specialist generating systematic diagnostic reasoning.
# Focus on {current_focus} while maintaining clinical rigor and evidence-based approach.
# Provide specific medical terminology, diagnostic criteria, and clinical decision-making rationale."""},
            
#             {"role": "user", "content": f"""{base_context}{previous_context}{exclusion_prompt}

# Generate a new diagnostic reasoning path focused on {current_focus}:
# 1. Use specific medical terminology and diagnostic criteria
# 2. Reference available medical knowledge appropriately  
# 3. Provide detailed clinical reasoning
# 4. Consider evidence-based medicine principles
# 5. Include confidence levels for diagnostic hypotheses

# Start with 'DIAGNOSTIC PATH #{i+1}:' and provide comprehensive medical reasoning."""}
#         ]
        
#         try:
#             response = client.chat.completions.create(
#                 model="gpt-4.1-2025-04-14",
#                 messages=messages,
#                 temperature=0.5 + i*0.1,  # Controlled temperature increase
#                 max_tokens=1500
#             )
            
#             thought = response.choices[0].message.content.strip()
#             thoughts.append(thought)
            
#             # Extract medical concepts for diversity tracking
#             medical_terms = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:syndrome|disease|disorder|infection|cancer|tumor|condition|diagnosis))?)\b', thought)
#             explored_concepts.update([term for term in medical_terms[:3] if len(term) > 3])
            
#         except Exception as e:
#             print(f"Error in sequential thought generation: {e}")
#             fallback_thought = f"""DIAGNOSTIC PATH #{i+1}:
# Focus: {current_focus}
# Clinical approach for: {input_text}
# Requires systematic evaluation integrating available medical knowledge.
# Consider evidence-based diagnostic criteria and clinical guidelines.
# Develop comprehensive assessment plan with appropriate testing strategy."""
#             thoughts.append(fallback_thought)
    
#     return thoughts

# # ================== Enhanced State Evaluator Methods ==================

# def evaluate_comprehensive(thoughts, input_text, medical_knowledge):
#     """Enhanced comprehensive medical evaluator with robust scoring"""
#     if not thoughts:
#         return [], []
    
#     evaluation_criteria = """
# Evaluate each diagnostic path based on medical standards (1-10 scale):
# 1. Clinical Reasoning Quality (3 points): Systematic approach, logical flow, medical accuracy
# 2. Diagnostic Accuracy (2 points): Appropriate differential diagnoses, evidence-based conclusions  
# 3. Knowledge Integration (2 points): Effective use of medical knowledge and clinical guidelines
# 4. Completeness (2 points): Comprehensive assessment covering key clinical areas
# 5. Practical Applicability (1 point): Realistic and implementable clinical approach

# Provide numerical score (1-10) and detailed medical justification.
# """
    
#     scores = []
#     justifications = []
    
#     for i, thought in enumerate(thoughts):
#         try:
#             messages = [
#                 {"role": "system", "content": f"""You are a senior medical expert evaluating diagnostic reasoning quality. 
# Use strict medical standards and evidence-based criteria. {evaluation_criteria}
# Score conservatively - only excellent medical reasoning should receive scores above 8."""},
                
#                 {"role": "user", "content": f"""Patient Case: {input_text}

# Available Medical Knowledge: {medical_knowledge}

# Evaluate this diagnostic path:
# {thought}

# Provide structured evaluation:
# 1. **NUMERICAL SCORE (1-10)**: [Provide single number]
# 2. **Clinical Reasoning Assessment**: [Detailed medical evaluation]
# 3. **Diagnostic Accuracy Review**: [Assessment of differential diagnoses]
# 4. **Knowledge Integration Analysis**: [How well medical knowledge was used]
# 5. **Areas for Improvement**: [Specific clinical recommendations]

# Be precise with scoring - use decimal points if needed (e.g., 7.5)."""}
#             ]
            
#             response = client.chat.completions.create(
#                 model="gpt-4.1-2025-04-14",
#                 messages=messages,
#                 temperature=0.1  # Low temperature for consistent evaluation
#             )
            
#             eval_content = response.choices[0].message.content
            
#             # Enhanced score extraction with multiple robust patterns
#             score_patterns = [
#                 r'(?:NUMERICAL SCORE|Score).*?[:\(]?\s*([0-9]+\.?[0-9]*)',
#                 r'Score\s*[:\-=]\s*([0-9]+\.?[0-9]*)',
#                 r'([0-9]+\.?[0-9]*)\s*(?:/10|out of 10)',
#                 r'Overall.*?([0-9]+\.?[0-9]*)',
#                 r'Total.*?([0-9]+\.?[0-9]*)'
#             ]
            
#             score = None
#             for pattern in score_patterns:
#                 score_match = re.search(pattern, eval_content, re.IGNORECASE)
#                 if score_match:
#                     try:
#                         potential_score = float(score_match.group(1))
#                         if 1.0 <= potential_score <= 10.0:
#                             score = potential_score
#                             break
#                     except ValueError:
#                         continue
            
#             # If no valid score found, analyze content for quality indicators
#             if score is None:
#                 quality_indicators = {
#                     'excellent': 8.5, 'outstanding': 9.0, 'superior': 8.0,
#                     'good': 7.0, 'adequate': 6.0, 'satisfactory': 6.5,
#                     'poor': 4.0, 'inadequate': 3.0, 'insufficient': 3.5,
#                     'comprehensive': 7.5, 'thorough': 7.0, 'systematic': 7.0
#                 }
                
#                 content_lower = eval_content.lower()
#                 for indicator, indicator_score in quality_indicators.items():
#                     if indicator in content_lower:
#                         score = indicator_score
#                         break
                
#                 # Final fallback based on content length and medical terms
#                 if score is None:
#                     medical_term_count = len(re.findall(r'\b(?:diagnosis|symptom|treatment|clinical|medical|patient|condition)\b', content_lower))
#                     if medical_term_count > 10:
#                         score = 6.5  # Reasonable default for medical content
#                     else:
#                         score = 5.0  # Conservative default
            
#             scores.append(score)
#             justifications.append(eval_content)
            
#         except Exception as e:
#             print(f"Error evaluating thought {i}: {e}")
#             # Assign conservative score for errors
#             scores.append(4.0)
#             justifications.append(f"Evaluation failed: {str(e)} - assigned conservative score")
    
#     return scores, justifications

# def evaluate_comparative_detailed(thoughts, input_text, medical_knowledge):
#     """Enhanced comparative evaluator with medical expertise and robust ranking"""
#     if not thoughts:
#         return [], []
    
#     try:
#         # Build comparison with numbered paths for clarity
#         comparison_text = "\n\n" + "="*50 + "\n\n".join([f"DIAGNOSTIC PATH {i+1}:\n{thought}" for i, thought in enumerate(thoughts)])
        
#         messages = [
#             {"role": "system", "content": """You are a senior attending physician comparing diagnostic approaches.
# Rank paths based on medical accuracy, clinical reasoning quality, and practical applicability.
# Use strict medical standards - be conservative with high scores."""},
            
#             {"role": "user", "content": f"""Patient Case: {input_text}

# Medical Knowledge Available: {medical_knowledge}

# Diagnostic Paths to Compare:
# {comparison_text}

# Please provide detailed comparative analysis:

# 1. **RANKING ORDER** (Best to Worst):
#    PATH X: Rank=1, Score=Y.Z
#    PATH X: Rank=2, Score=Y.Z
#    [Continue for all paths]

# 2. **DETAILED JUSTIFICATION**:
#    - Medical accuracy assessment
#    - Clinical reasoning evaluation  
#    - Knowledge integration review
#    - Practical applicability analysis

# 3. **TOP PERFORMER ANALYSIS**: Why is the highest-ranked path superior?

# Use decimal scores (1.0-10.0) with conservative medical standards."""}
#         ]
        
#         response = client.chat.completions.create(
#             model="gpt-4.1-2025-04-14",
#             messages=messages,
#             temperature=0.1
#         )
        
#         eval_content = response.choices[0].message.content
        
#         # Enhanced score parsing with multiple strategies
#         scores = [5.0] * len(thoughts)  # Conservative defaults
        
#         # Strategy 1: Parse structured ranking format
#         ranking_pattern = r'PATH\s+(\d+):\s*(?:Rank=\d+,?)?\s*Score=([0-9]+\.?[0-9]*)'
#         ranking_matches = re.findall(ranking_pattern, eval_content, re.IGNORECASE)
        
#         for path_num_str, score_str in ranking_matches:
#             try:
#                 path_idx = int(path_num_str) - 1  # Convert to 0-based index
#                 score = float(score_str)
#                 if 0 <= path_idx < len(thoughts) and 1.0 <= score <= 10.0:
#                     scores[path_idx] = score
#             except (ValueError, IndexError):
#                 continue
        
#         # Strategy 2: Look for individual path scores if structured format failed
#         if all(score == 5.0 for score in scores):
#             for i in range(len(thoughts)):
#                 path_patterns = [
#                     f'PATH\\s+{i+1}.*?([0-9]+\\.?[0-9]*)(?:/10|\\s+out of 10)',
#                     f'Path\\s+{i+1}.*?Score.*?([0-9]+\\.?[0-9]*)',
#                     f'{i+1}[\\)\\.].*?([0-9]+\\.?[0-9]*)'
#                 ]
                
#                 for pattern in path_patterns:
#                     match = re.search(pattern, eval_content, re.IGNORECASE)
#                     if match:
#                         try:
#                             score = float(match.group(1))
#                             if 1.0 <= score <= 10.0:
#                                 scores[i] = score
#                                 break
#                         except ValueError:
#                             continue
        
#         # Strategy 3: Content quality analysis if scores still default
#         remaining_defaults = [i for i, score in enumerate(scores) if score == 5.0]
#         if remaining_defaults:
#             quality_keywords = {
#                 'excellent': 8.0, 'outstanding': 8.5, 'superior': 7.5,
#                 'good': 6.5, 'adequate': 6.0, 'comprehensive': 7.0,
#                 'poor': 4.0, 'inadequate': 3.5, 'insufficient': 4.0
#             }
            
#             for idx in remaining_defaults:
#                 path_section = eval_content[eval_content.find(f"PATH {idx+1}"):eval_content.find(f"PATH {idx+2}") if idx+1 < len(thoughts) else len(eval_content)]
                
#                 for keyword, keyword_score in quality_keywords.items():
#                     if keyword.lower() in path_section.lower():
#                         scores[idx] = keyword_score
#                         break
        
#         return scores, eval_content
        
#     except Exception as e:
#         print(f"Error in comparative evaluation: {e}")
#         # Return conservative scores with explanation
#         conservative_scores = [5.5 + random.uniform(-0.5, 0.5) for _ in thoughts]  # Slight variation around median
#         return conservative_scores, f"Evaluation failed: {str(e)} - assigned conservative scores"

# # ================== Enhanced Search Algorithms ==================

# def tot_breadth_first_search(input_text, medical_knowledge, thought_generator, evaluator,
#                            step_limit=4, breadth=3, samples_per_step=4):
#     """Enhanced medical BFS search with improved state management and medical focus"""
#     print(f"Starting Enhanced Medical BFS: {step_limit} steps, breadth={breadth}")
    
#     # Initial state with medical context
#     states = [{
#         "context": f"Clinical Case: {input_text}",
#         "path": [],
#         "score": 6.0,  # Neutral starting score
#         "confidence": 0.5,
#         "focus_areas": ["initial_assessment"],
#         "medical_concepts": set()
#     }]
    
#     all_step_scores = []
    
#     for step in range(step_limit):
#         print(f"  Step {step+1}/{step_limit}: Processing {len(states)} states...")
#         next_states = []
        
#         # Define medical focus progression
#         medical_focus_progression = [
#             "differential_diagnosis",
#             "diagnostic_testing", 
#             "treatment_planning",
#             "prognosis_monitoring"
#         ]
#         current_focus = medical_focus_progression[step % len(medical_focus_progression)]
        
#         for state in states:
#             # Build enhanced medical context
#             context_parts = [f"Clinical Case: {input_text}"]
            
#             if state['path']:
#                 context_parts.append("Previous Clinical Reasoning:")
#                 for i, path_step in enumerate(state['path']):
#                     # Truncate long steps for context management
#                     step_summary = path_step[:200] + "..." if len(path_step) > 200 else path_step
#                     context_parts.append(f"Step {i+1}: {step_summary}")
            
#             context_parts.append(f"Current Focus: {current_focus}")
#             context_parts.append(f"Medical Knowledge: {medical_knowledge}")
            
#             full_context = "\n\n".join(context_parts)
            
#             # Generate new thoughts with medical focus
#             try:
#                 if hasattr(thought_generator, '__name__') and 'sampling' in thought_generator.__name__:
#                     new_thoughts = thought_generator(full_context, medical_knowledge, samples_per_step, current_focus)
#                 else:
#                     new_thoughts = thought_generator(full_context, medical_knowledge, samples_per_step)
#             except Exception as e:
#                 print(f"Error generating thoughts: {e}")
#                 new_thoughts = [f"Clinical reasoning step {step+1} for case: {input_text}"]
            
#             # Create enhanced states for each new thought
#             for thought in new_thoughts:
#                 if thought and len(thought.strip()) > 20:  # Quality filter
#                     new_path = state["path"].copy()
#                     new_path.append(thought.strip())
                    
#                     # Extract medical concepts for tracking
#                     medical_terms = set(re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:syndrome|disease|disorder|infection|condition))?)\b', thought))
                    
#                     next_states.append({
#                         "context": thought,
#                         "path": new_path,
#                         "score": 0,
#                         "confidence": 0,
#                         "focus_areas": state["focus_areas"] + [current_focus],
#                         "medical_concepts": state["medical_concepts"].union(medical_terms)
#                     })
        
#         # Evaluate all new states with robust error handling
#         if next_states:
#             thought_texts = [s["context"] for s in next_states]
            
#             try:
#                 # Check evaluator signature and call appropriately
#                 import inspect
#                 sig = inspect.signature(evaluator)
                
#                 if len(sig.parameters) >= 3:
#                     scores, justifications = evaluator(thought_texts, input_text, medical_knowledge)
#                 else:
#                     result = evaluator(thought_texts)
#                     if isinstance(result, tuple):
#                         scores, justifications = result
#                     else:
#                         scores = result
#                         justifications = [""] * len(scores)
                
#                 # Ensure scores are valid
#                 if not isinstance(scores, list) or len(scores) != len(next_states):
#                     scores = [6.0] * len(next_states)  # Safe fallback
                    
#             except Exception as e:
#                 print(f"Error in evaluation: {e}")
#                 scores = [6.0] * len(next_states)  # Conservative fallback
            
#             # Update states with scores and confidence
#             for i, score in enumerate(scores):
#                 if i < len(next_states):
#                     # Ensure score is within valid range
#                     validated_score = max(1.0, min(10.0, float(score)))
#                     next_states[i]["score"] = validated_score
#                     next_states[i]["confidence"] = min(1.0, validated_score / 10.0)
            
#             all_step_scores.append(scores[:len(next_states)])
        
#         # Select best states with diversity consideration
#         next_states.sort(key=lambda x: (x["score"], len(x["medical_concepts"])), reverse=True)
        
#         # Apply diversity filter to avoid similar paths
#         selected_states = []
#         seen_concepts = set()
        
#         for state in next_states:
#             if len(selected_states) >= breadth:
#                 break
                
#             # Check concept diversity
#             state_concepts = state["medical_concepts"]
#             concept_overlap = len(state_concepts.intersection(seen_concepts))
            
#             if concept_overlap < len(state_concepts) * 0.7:  # Allow up to 70% overlap
#                 selected_states.append(state)
#                 seen_concepts.update(state_concepts)
        
#         # Fill remaining slots if needed
#         while len(selected_states) < breadth and len(selected_states) < len(next_states):
#             for state in next_states:
#                 if state not in selected_states:
#                     selected_states.append(state)
#                     break
        
#         states = selected_states
        
#         if states:
#             best_scores = [f'{s["score"]:.1f}' for s in sorted(states, key=lambda x: x["score"], reverse=True)]
#             print(f"  Best scores this step: {best_scores}")
    
#     # Return best path with comprehensive information
#     if states:
#         best_state = max(states, key=lambda x: x["score"])
#         return best_state["path"], all_step_scores
#     return [], all_step_scores

# def tot_depth_first_search(input_text, medical_knowledge, thought_generator, evaluator,
#                          step_limit=4, threshold=3.0, samples_per_step=3, max_paths=5):
#     """Enhanced medical DFS search with improved pruning and medical focus"""
#     print(f"Starting Enhanced Medical DFS: threshold={threshold}")
    
#     final_paths = []
#     explored_count = 0
#     max_explorations = 30  # Increased for better exploration
    
#     # Medical focus areas for systematic exploration
#     medical_focus_areas = ["symptom_analysis", "differential_diagnosis", "diagnostic_testing", "treatment_planning"]
    
#     def dfs_recursive(current_context, depth=0, current_path=None, path_scores=None, 
#                      focus_history=None, explored_concepts=None):
#         nonlocal explored_count
        
#         if explored_count >= max_explorations or len(final_paths) >= max_paths:
#             return
        
#         # Initialize parameters
#         if current_path is None:
#             current_path = []
#         if path_scores is None:
#             path_scores = []
#         if focus_history is None:
#             focus_history = []
#         if explored_concepts is None:
#             explored_concepts = set()
        
#         explored_count += 1
        
#         # Termination condition with quality check
#         if depth >= step_limit:
#             if path_scores and len(path_scores) > 0:
#                 avg_score = sum(path_scores) / len(path_scores)
#                 if avg_score >= threshold:  # Only keep high-quality paths
#                     final_paths.append((current_path.copy(), avg_score, path_scores.copy()))
#             return
        
#         # Determine medical focus for current depth
#         current_focus = medical_focus_areas[depth % len(medical_focus_areas)]
        
#         # Build enhanced context with medical progression
#         context_parts = [
#             f"Clinical Case: {input_text}",
#             f"Medical Knowledge: {medical_knowledge}",
#             f"Current Focus: {current_focus}",
#             f"Depth: {depth+1}/{step_limit}"
#         ]
        
#         if current_path:
#             context_parts.append("Previous Clinical Steps:")
#             for i, step in enumerate(current_path):
#                 step_summary = step[:150] + "..." if len(step) > 150 else step
#                 context_parts.append(f"Step {i+1}: {step_summary}")
        
#         if explored_concepts:
#             context_parts.append(f"Explored Concepts: {', '.join(list(explored_concepts)[:5])}")
        
#         full_context = "\n\n".join(context_parts)
        
#         # Generate thoughts with error handling
#         try:
#             if hasattr(thought_generator, '__name__') and 'sampling' in thought_generator.__name__:
#                 new_thoughts = thought_generator(full_context, medical_knowledge, samples_per_step, current_focus)
#             else:
#                 new_thoughts = thought_generator(full_context, medical_knowledge, samples_per_step)
#         except Exception as e:
#             print(f"Error generating thoughts at depth {depth}: {e}")
#             return
        
#         # Filter out low-quality thoughts
#         quality_thoughts = [t for t in new_thoughts if t and len(t.strip()) > 50]
#         if not quality_thoughts:
#             return
        
#         # Evaluate thoughts with robust error handling
#         try:
#             import inspect
#             sig = inspect.signature(evaluator)
            
#             if len(sig.parameters) >= 3:
#                 scores, _ = evaluator(quality_thoughts, input_text, medical_knowledge)
#             else:
#                 result = evaluator(quality_thoughts)
#                 if isinstance(result, tuple):
#                     scores, _ = result
#                 else:
#                     scores = result
            
#             # Validate scores
#             if not isinstance(scores, list) or len(scores) != len(quality_thoughts):
#                 scores = [6.0] * len(quality_thoughts)
                
#         except Exception as e:
#             print(f"Error evaluating thoughts at depth {depth}: {e}")
#             scores = [6.0] * len(quality_thoughts)
        
#         # Sort by score and explore high-quality paths
#         scored_thoughts = list(zip(quality_thoughts, scores))
#         scored_thoughts.sort(key=lambda x: x[1], reverse=True)
        
#         # Explore promising thoughts with diversity consideration
#         for thought, score in scored_thoughts:
#             if len(final_paths) >= max_paths:
#                 break
                
#             # Extract medical concepts for diversity
#             medical_terms = set(re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:syndrome|disease|disorder|infection|condition))?)\b', thought))
            
#             # Check for concept diversity or high quality
#             concept_overlap = len(medical_terms.intersection(explored_concepts))
#             is_diverse = concept_overlap < len(medical_terms) * 0.6
#             is_high_quality = score >= threshold
            
#             if is_high_quality and (is_diverse or depth == 0):
#                 new_path = current_path.copy()
#                 new_path.append(thought.strip())
#                 new_scores = path_scores.copy()
#                 new_scores.append(score)
#                 new_focus_history = focus_history.copy()
#                 new_focus_history.append(current_focus)
#                 new_concepts = explored_concepts.union(medical_terms)
                
#                 dfs_recursive(thought, depth + 1, new_path, new_scores, 
#                             new_focus_history, new_concepts)
    
#     # Start recursive search
#     dfs_recursive(input_text)
    
#     # Return best path
#     if final_paths:
#         final_paths.sort(key=lambda x: x[1], reverse=True)
#         best_path, best_avg_score, best_score_history = final_paths[0]
#         print(f"DFS found {len(final_paths)} valid paths, best average score: {best_avg_score:.2f}")
#         return best_path, [best_score_history]
    
#     print("DFS found no paths meeting threshold criteria")
#     return [], []


# import re
# import time
# import random
# from typing import List, Dict, Any, Callable, Tuple, Optional
# import numpy as np

# # ================== Enhanced Answer Extraction ==================
# def extract_answer_from_paths(reasoning_paths, input_text, medical_knowledge):
#     """
#     Enhanced medical answer extraction with improved synthesis and quality assessment
#     """
#     if not reasoning_paths:
#         return "Unable to generate diagnosis due to insufficient reasoning paths."
    
#     # Analyze path quality and diversity
#     path_quality_scores = []
#     medical_concepts = set()
    
#     for path in reasoning_paths:
#         # Extract medical terminology density
#         medical_terms = re.findall(r'\b(?:diagnosis|symptom|syndrome|disease|disorder|treatment|therapy|medication|test|examination|clinical|patient|condition|pathology)\b', path.lower())
#         medical_density = len(medical_terms) / max(len(path.split()), 1)
        
#         # Extract specific medical concepts
#         concepts = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:syndrome|disease|disorder|infection|cancer|tumor|condition|diagnosis))?)\b', path)
#         medical_concepts.update(concepts[:5])  # Top 5 concepts per path
        
#         # Assess reasoning structure (presence of logical flow)
#         structure_indicators = ['because', 'therefore', 'indicates', 'suggests', 'evidence', 'based on', 'considering']
#         structure_score = sum(1 for indicator in structure_indicators if indicator in path.lower())
        
#         # Combined quality score
#         quality = min(10.0, medical_density * 20 + structure_score * 0.5 + len(concepts) * 0.3)
#         path_quality_scores.append(quality)
    
#     # Weight paths by quality for synthesis
#     total_quality = sum(path_quality_scores)
#     path_weights = [score / max(total_quality, 1) for score in path_quality_scores]
    
#     # Create comprehensive context for synthesis
#     weighted_reasoning = ""
#     for i, (path, weight) in enumerate(zip(reasoning_paths, path_weights)):
#         importance = "High" if weight > 0.3 else "Moderate" if weight > 0.15 else "Supporting"
#         weighted_reasoning += f"\n\n[{importance} Priority] Reasoning Path {i+1} (Weight: {weight:.2f}):\n{path}"
    
#     try:
#         messages = [
#             {"role": "system", "content": """You are a senior medical expert providing comprehensive clinical assessment. 
#             Synthesize multiple diagnostic reasoning paths into a cohesive, evidence-based medical conclusion.
            
#             Requirements:
#             1. Prioritize higher-weighted reasoning paths in your synthesis
#             2. Identify convergent diagnostic patterns across paths
#             3. Address any conflicting assessments with clinical reasoning
#             4. Provide actionable clinical recommendations
#             5. Include confidence levels based on evidence strength"""},
            
#             {"role": "user", "content": f"""Patient Case: {input_text}

# Available Medical Knowledge: {medical_knowledge}

# Quality-Weighted Diagnostic Reasoning Paths:
# {weighted_reasoning}

# Key Medical Concepts Identified: {', '.join(list(medical_concepts)[:10])}

# Please provide a comprehensive medical assessment including:

# 1. **PRIMARY DIAGNOSIS** (with confidence level 1-10)
#    - Most likely diagnosis based on convergent evidence
#    - Clinical justification from multiple reasoning paths

# 2. **DIFFERENTIAL DIAGNOSES** (ranked by likelihood)
#    - Alternative diagnoses with brief rationale
#    - Distinguishing clinical features

# 3. **DIAGNOSTIC WORKUP**
#    - Essential tests/examinations needed
#    - Rationale for each recommendation

# 4. **MANAGEMENT PLAN**
#    - Immediate interventions if indicated
#    - Follow-up recommendations

# 5. **CLINICAL REASONING SYNTHESIS**
#    - How the multiple reasoning paths support the conclusion
#    - Areas of convergence and any conflicting evidence

# Format your response clearly with confidence levels and clinical reasoning."""}
#         ]
        
#         response = client.chat.completions.create(
#             model="gpt-4.1-2025-04-14",
#             messages=messages,
#             temperature=0.1,  # Lower temperature for more consistent medical reasoning
#             max_tokens=2000
#         )
        
#         synthesis = response.choices[0].message.content
        
#         # Add quality metrics to the final answer
#         quality_summary = f"""

# --- REASONING QUALITY METRICS ---
# • Total Reasoning Paths Analyzed: {len(reasoning_paths)}
# • Average Path Quality Score: {np.mean(path_quality_scores):.2f}/10.0
# • Medical Concept Diversity: {len(medical_concepts)} unique concepts
# • Highest Quality Path Weight: {max(path_weights):.2f}
# • Evidence Convergence: {'High' if max(path_weights) < 0.6 else 'Moderate' if max(path_weights) < 0.8 else 'Low'}"""
        
#         return synthesis + quality_summary
        
#     except Exception as e:
#         print(f"Error generating comprehensive answer: {e}")
#         # Provide structured fallback with available information
#         fallback_answer = f"""
# **CLINICAL ASSESSMENT SUMMARY**

# Primary Case: {input_text}

# Based on {len(reasoning_paths)} reasoning paths analyzed:
# - Average reasoning quality: {np.mean(path_quality_scores):.1f}/10.0
# - Key medical concepts identified: {', '.join(list(medical_concepts)[:5])}

# **RECOMMENDATION**: The clinical assessment suggests multiple diagnostic possibilities require systematic evaluation. Consider comprehensive history, physical examination, and appropriate diagnostic testing based on the presenting symptoms and available medical knowledge.

# **CLINICAL REASONING**: Multiple diagnostic approaches were analyzed with varying quality scores. Further clinical correlation and evidence-based assessment recommended.

# Error in detailed synthesis: {str(e)}
# """
#         return fallback_answer

# # ================== Enhanced ToTResult Class ==================
# class ToTResult:
#     """
#     Enhanced data class to store Tree of Thoughts results with comprehensive metrics
#     """
#     def __init__(self, method_name, reasoning_paths, final_answer, execution_time, 
#                  confidence_score, step_scores, total_api_calls, quality_metrics=None):
#         self.method_name = method_name
#         self.reasoning_paths = reasoning_paths
#         self.final_answer = final_answer
#         self.execution_time = execution_time
#         self.confidence_score = confidence_score
#         self.step_scores = step_scores
#         self.total_api_calls = total_api_calls
#         self.quality_metrics = quality_metrics or {}
        
#         # Compute additional derived metrics
#         self._compute_derived_metrics()
    
#     def _compute_derived_metrics(self):
#         """Compute additional quality and performance metrics"""
#         # Path quality analysis
#         if self.reasoning_paths:
#             path_lengths = [len(path.split()) for path in self.reasoning_paths]
#             self.quality_metrics.update({
#                 'avg_path_length': np.mean(path_lengths),
#                 'path_length_std': np.std(path_lengths),
#                 'total_paths': len(self.reasoning_paths),
#                 'path_diversity': len(set(path[:100] for path in self.reasoning_paths)) / len(self.reasoning_paths)
#             })
        
#         # Score analysis
#         if self.step_scores:
#             all_scores = []
#             for step in self.step_scores:
#                 if isinstance(step, list):
#                     all_scores.extend(step)
#                 else:
#                     all_scores.append(step)
            
#             if all_scores:
#                 self.quality_metrics.update({
#                     'score_mean': np.mean(all_scores),
#                     'score_std': np.std(all_scores),
#                     'score_min': min(all_scores),
#                     'score_max': max(all_scores),
#                     'score_stability': 1.0 - (np.std(all_scores) / max(np.mean(all_scores), 1))
#                 })
        
#         # Performance efficiency
#         if self.execution_time > 0 and self.total_api_calls > 0:
#             self.quality_metrics.update({
#                 'efficiency_score': len(self.reasoning_paths) / max(self.execution_time, 0.1),
#                 'api_efficiency': len(self.reasoning_paths) / max(self.total_api_calls, 1)
#             })
    
#     def get_overall_quality_score(self):
#         """Calculate comprehensive quality score combining multiple metrics"""
#         base_confidence = self.confidence_score
        
#         # Adjust based on path quality
#         path_quality_bonus = 0
#         if 'path_diversity' in self.quality_metrics:
#             path_quality_bonus += self.quality_metrics['path_diversity'] * 0.1
        
#         # Adjust based on score stability
#         stability_bonus = 0
#         if 'score_stability' in self.quality_metrics:
#             stability_bonus += self.quality_metrics['score_stability'] * 0.1
        
#         # Efficiency consideration
#         efficiency_bonus = 0
#         if 'efficiency_score' in self.quality_metrics and self.quality_metrics['efficiency_score'] > 1:
#             efficiency_bonus += min(0.05, self.quality_metrics['efficiency_score'] * 0.01)
        
#         overall_score = base_confidence + path_quality_bonus + stability_bonus + efficiency_bonus
#         return min(1.0, max(0.0, overall_score))
    
#     def get_summary_dict(self):
#         """Return summary dictionary for easy comparison"""
#         return {
#             'method': self.method_name,
#             'confidence': self.confidence_score,
#             'overall_quality': self.get_overall_quality_score(),
#             'execution_time': self.execution_time,
#             'num_paths': len(self.reasoning_paths),
#             'api_calls': self.total_api_calls,
#             **self.quality_metrics
#         }

# # ================== Enhanced Confidence Score Calculation ==================
# def calculate_confidence_score(reasoning_paths, step_scores, method_name=None):
#     """
#     Enhanced confidence score calculation with multiple quality indicators
#     """
#     if not step_scores or not reasoning_paths:
#         return 0.1  # Very low confidence for empty results
    
#     # 1. Score-based confidence
#     all_scores = []
#     for step in step_scores:
#         if isinstance(step, list):
#             all_scores.extend([s for s in step if isinstance(s, (int, float))])
#         elif isinstance(step, (int, float)):
#             all_scores.append(step)
    
#     if not all_scores:
#         return 0.2  # Low confidence for missing scores
    
#     # Base confidence from average scores (normalized to 0-1)
#     avg_score = np.mean(all_scores)
#     base_confidence = min(1.0, max(0.0, (avg_score - 1.0) / 9.0))  # Scale 1-10 to 0-1
    
#     # 2. Path quality indicators
#     path_quality_factors = []
    
#     for path in reasoning_paths:
#         if not path or len(path.strip()) < 20:
#             path_quality_factors.append(0.1)
#             continue
            
#         # Medical terminology density
#         medical_terms = len(re.findall(r'\b(?:diagnosis|symptom|syndrome|disease|disorder|treatment|clinical|patient|condition)\b', path.lower()))
#         medical_density = medical_terms / max(len(path.split()), 1)
        
#         # Reasoning structure indicators
#         reasoning_indicators = ['because', 'therefore', 'evidence', 'indicates', 'suggests', 'based on', 'considering', 'due to']
#         reasoning_score = sum(1 for indicator in reasoning_indicators if indicator in path.lower()) / len(reasoning_indicators)
        
#         # Length appropriateness (not too short, not excessively long)
#         length_words = len(path.split())
#         length_score = 1.0 if 50 <= length_words <= 500 else max(0.3, min(length_words / 200, 500 / length_words))
        
#         # Combined path quality
#         path_quality = (medical_density * 0.4 + reasoning_score * 0.4 + length_score * 0.2)
#         path_quality_factors.append(min(1.0, path_quality))
    
#     avg_path_quality = np.mean(path_quality_factors) if path_quality_factors else 0.3
    
#     # 3. Score consistency (penalize high variance)
#     score_consistency = 1.0
#     if len(all_scores) > 1:
#         score_std = np.std(all_scores)
#         score_consistency = max(0.3, 1.0 - (score_std / max(np.mean(all_scores), 1)) * 0.5)
    
#     # 4. Method-specific adjustments
#     method_bonus = 0.0
#     if method_name:
#         # BFS methods tend to be more comprehensive
#         if 'BFS' in method_name:
#             method_bonus += 0.05
#         # Sequential methods tend to be more systematic
#         if 'Sequential' in method_name:
#             method_bonus += 0.03
#         # Comprehensive evaluation tends to be more thorough
#         if 'Comprehensive' in method_name:
#             method_bonus += 0.02
    
#     # 5. Combine all factors
#     final_confidence = (
#         base_confidence * 0.4 +           # 40% from scores
#         avg_path_quality * 0.35 +         # 35% from path quality
#         score_consistency * 0.20 +        # 20% from consistency
#         method_bonus                      # 5% method bonus
#     )
    
#     # Ensure reasonable bounds with some variance
#     final_confidence = min(0.95, max(0.15, final_confidence))
    
#     # Add small random variation to avoid identical scores (±2%)
#     variation = random.uniform(-0.02, 0.02)
#     final_confidence = min(0.98, max(0.12, final_confidence + variation))
    
#     return round(final_confidence, 3)

# # ================== Enhanced Benchmark Runner ==================
# def run_tot_benchmark(input_text, medical_knowledge, test_methods=None):
#     """
#     Enhanced medical ToT benchmark with improved result analysis and ranking
#     """
    
#     if test_methods is None:
#         test_methods = [
#             "Sampling_Comprehensive_BFS",
#             "Sampling_Comprehensive_DFS", 
#             "Sampling_Comparative_BFS",
#             "Sampling_Comparative_DFS",
#             "Sequential_Comprehensive_BFS",
#             "Sequential_Comprehensive_DFS",
#             "Sequential_Comparative_BFS",
#             "Sequential_Comparative_DFS"
#         ]
    
#     # Method mappings (assuming these functions exist)
#     thought_generators = {
#         "Sampling": thought_generator_sampling,
#         "Sequential": thought_generator_sequential
#     }
    
#     evaluators = {
#         "Comprehensive": evaluate_comprehensive,
#         "Comparative": evaluate_comparative_detailed
#     }
    
#     search_algorithms = {
#         "BFS": tot_breadth_first_search,
#         "DFS": tot_depth_first_search
#     }
    
#     # Enhanced result storage
#     benchmark_results = {}
#     api_call_count = 0
    
#     print(f"Starting Enhanced Medical ToT Benchmark")
#     print(f"Testing {len(test_methods)} method combinations")
#     print("="*60)
    
#     for method_combo in test_methods:
#         parts = method_combo.split("_")
#         if len(parts) != 3:
#             print(f"Skipping invalid method format: {method_combo}")
#             continue
        
#         gen_name, eval_name, search_name = parts
#         print(f"\n--- Testing {method_combo} ---")
        
#         # Get methods with validation
#         generator = thought_generators.get(gen_name)
#         evaluator = evaluators.get(eval_name)
#         search_algo = search_algorithms.get(search_name)
        
#         if not all([generator, evaluator, search_algo]):
#             print(f"Warning: Could not find methods for {method_combo}")
#             continue
        
#         start_time = time.time()
#         api_calls_before = api_call_count
        
#         try:
#             # Execute search with method-specific parameters
#             if search_name == "BFS":
#                 # BFS parameters - balance breadth and depth
#                 step_limit = 3 if 'Comprehensive' in eval_name else 2
#                 breadth = 3 if 'Sequential' in gen_name else 2
#                 samples_per_step = 4 if 'Comprehensive' in eval_name else 3
                
#                 reasoning_paths, step_scores = search_algo(
#                     input_text, medical_knowledge, generator, evaluator,
#                     step_limit=step_limit, breadth=breadth, samples_per_step=samples_per_step
#                 )
#             else:  # DFS
#                 # DFS parameters - focus on quality thresholds
#                 step_limit = 4 if 'Comprehensive' in eval_name else 3
#                 threshold = 6.5 if 'Comparative' in eval_name else 3.0
#                 samples_per_step = 4 if 'Sequential' in gen_name else 3
                
#                 reasoning_paths, step_scores = search_algo(
#                     input_text, medical_knowledge, generator, evaluator,
#                     step_limit=step_limit, threshold=threshold, samples_per_step=samples_per_step
#                 )
            
#             execution_time = time.time() - start_time
#             current_api_calls = api_call_count - api_calls_before
            
#             # Generate final answer
#             final_answer = extract_answer_from_paths(reasoning_paths, input_text, medical_knowledge)
            
#             # Calculate enhanced confidence score
#             confidence_score = calculate_confidence_score(reasoning_paths, step_scores, method_combo)
            
#             # Create ToTResult object with quality metrics
#             quality_metrics = {
#                 'method_type': f"{gen_name}_{eval_name}_{search_name}",
#                 'generator_type': gen_name,
#                 'evaluator_type': eval_name,
#                 'search_type': search_name
#             }
            
#             result = ToTResult(
#                 method_name=method_combo,
#                 reasoning_paths=reasoning_paths,
#                 final_answer=final_answer,
#                 execution_time=execution_time,
#                 confidence_score=confidence_score,
#                 step_scores=step_scores,
#                 total_api_calls=current_api_calls,
#                 quality_metrics=quality_metrics
#             )
            
#             benchmark_results[method_combo] = result
            
#             print(f"✓ Completed in {execution_time:.2f}s")
#             print(f"  Confidence: {confidence_score:.3f}")
#             print(f"  Overall Quality: {result.get_overall_quality_score():.3f}")
#             print(f"  Paths Generated: {len(reasoning_paths)}")
            
#         except Exception as e:
#             print(f"✗ Error in {method_combo}: {str(e)}")
#             # Create error result
#             error_result = ToTResult(
#                 method_name=method_combo,
#                 reasoning_paths=[],
#                 final_answer=f"Error: {str(e)}",
#                 execution_time=0.0,
#                 confidence_score=0.05,  # Very low confidence for errors
#                 step_scores=[],
#                 total_api_calls=0,
#                 quality_metrics={'error': True}
#             )
#             benchmark_results[method_combo] = error_result
    
#     # Print benchmark summary
#     print("\n" + "="*60)
#     print("BENCHMARK SUMMARY")
#     print("="*60)
    
#     if benchmark_results:
#         # Sort by overall quality score
#         sorted_results = sorted(
#             benchmark_results.items(), 
#             key=lambda x: x[1].get_overall_quality_score(), 
#             reverse=True
#         )
        
#         print("\nRanking by Overall Quality Score:")
#         for i, (method, result) in enumerate(sorted_results, 1):
#             print(f"{i:2d}. {method}: {result.get_overall_quality_score():.3f} "
#                   f"(Confidence: {result.confidence_score:.3f}, "
#                   f"Time: {result.execution_time:.1f}s)")
    
#     return benchmark_results

# # ================== Enhanced Integration Function ==================
# def generate_with_tree_of_thoughts(input_text, response_of_KG_list_path, response_of_KG_neighbor, method="all"):
#     """
#     Enhanced Tree of Thoughts generation with improved method handling and result quality
#     """
    
#     # Combine and validate knowledge sources
#     medical_knowledge = f"{response_of_KG_list_path}\n\n{response_of_KG_neighbor}".strip()
    
#     if not medical_knowledge or len(medical_knowledge) < 10:
#         print("Warning: Limited medical knowledge provided. Results may be suboptimal.")
#         medical_knowledge = "General medical knowledge for clinical assessment."
    
#     if method.lower() == "all":
#         # Run comprehensive benchmark
#         print("Running comprehensive Tree of Thoughts benchmark...")
#         benchmark_results = run_tot_benchmark(input_text, medical_knowledge)
        
#         # Convert ToTResult objects to dictionaries for easier handling
#         result_dicts = {}
#         for method_name, tot_result in benchmark_results.items():
#             if isinstance(tot_result, ToTResult):
#                 result_dicts[method_name] = tot_result.get_summary_dict()
#                 result_dicts[method_name]['final_answer'] = tot_result.final_answer
#                 result_dicts[method_name]['reasoning_paths'] = tot_result.reasoning_paths
#             else:
#                 # Handle legacy dictionary format
#                 result_dicts[method_name] = tot_result
        
#         return result_dicts
    
#     else:
#         # Handle specific method request
#         print(f"Running specific method: {method}")
        
#         # Parse method with enhanced validation
#         if '-' in method:
#             parts = method.split("-")
#         elif '_' in method:
#             parts = method.split("_")
#         else:
#             print(f"Invalid method format: {method}. Using default.")
#             parts = ["Sampling", "Comprehensive", "BFS"]
        
#         if len(parts) != 3:
#             print(f"Invalid method format: {method}. Using default Sampling-Comprehensive-BFS.")
#             parts = ["Sampling", "Comprehensive", "BFS"]
        
#         gen_name, eval_name, search_name = [p.capitalize() for p in parts]
        
#         # Get method functions with fallbacks
#         generators = {
#             "Sampling": thought_generator_sampling, 
#             "Sequential": thought_generator_sequential
#         }
#         evaluators = {
#             "Comprehensive": evaluate_comprehensive, 
#             "Comparative": evaluate_comparative_detailed
#         }
#         search_algos = {
#             "BFS": tot_breadth_first_search, 
#             "DFS": tot_depth_first_search
#         }
        
#         generator = generators.get(gen_name, thought_generator_sampling)
#         evaluator = evaluators.get(eval_name, evaluate_comprehensive)  
#         search_algo = search_algos.get(search_name, tot_breadth_first_search)
        
#         # Execute specific method with optimized parameters
#         try:
#             start_time = time.time()
            
#             if search_name == "BFS":
#                 reasoning_paths, step_scores = search_algo(
#                     input_text, medical_knowledge, generator, evaluator,
#                     step_limit=3, breadth=2, samples_per_step=3
#                 )
#             else:  # DFS
#                 reasoning_paths, step_scores = search_algo(
#                     input_text, medical_knowledge, generator, evaluator,
#                     step_limit=3, threshold=6.0, samples_per_step=3
#                 )
            
#             execution_time = time.time() - start_time
            
#             # Generate final answer and calculate metrics
#             final_answer = extract_answer_from_paths(reasoning_paths, input_text, medical_knowledge)
#             confidence_score = calculate_confidence_score(reasoning_paths, step_scores, method)
            
#             # Return structured result
#             return {
#                 'final_answer': final_answer,
#                 'reasoning_paths': reasoning_paths,
#                 'confidence_score': confidence_score,
#                 'execution_time': execution_time,
#                 'method_used': method,
#                 'step_scores': step_scores
#             }, reasoning_paths
            
#         except Exception as e:
#             print(f"Error executing method {method}: {e}")
#             fallback_answer = f"Error in Tree of Thoughts processing: {str(e)}"
#             return fallback_answer, []
# #--------------------------------------


###-------------------tot ——————————————————————————

import asyncio
import re
import time
import random
from typing import List, Dict, Any, Callable, Tuple, Optional
import numpy as np
from openai import AsyncOpenAI # Make sure this is imported
import os # For OPENAI_API_KEY

# --- Tree of Thoughts (ToT) Refactored ASYNC Functions ---

async def thought_generator_sampling(async_client_param: AsyncOpenAI, input_text: str, medical_knowledge: str, num_samples: int = 3, focus_area: Optional[str] = None) -> List[str]:
    """Enhanced medical thought sampling generator with better structure (ASYNC)"""
    focus_prompt_content = f"Pay special attention to {focus_area} aspects." if focus_area else ""
    
    messages = [
        {"role": "system", "content": f"""You are a professional medical AI assistant. Generate diverse, high-quality diagnostic reasoning paths.
Each path should follow structured medical reasoning: Chief Complaint → History → Physical Exam → Differential Diagnosis → Tests → Final Diagnosis.
{focus_prompt_content} Ensure each path explores different possibilities and reasoning approaches.
Be specific about medical conditions, diagnostic criteria, and clinical decision-making. Each path must start with 'DIAGNOSTIC PATH #X:'"""},
        {"role": "user", "content": f"""Patient Case: {input_text}
Available Medical Knowledge (summary): {medical_knowledge[:1000]}...
Generate {num_samples} distinct diagnostic reasoning paths following the instructions."""}
    ]
    
    try:
        response = await async_client_param.chat.completions.create(
            model="gpt-4o-mini", # Or your preferred model, e.g., "gpt-4.1-2025-04-14"
            messages=messages,
            temperature=0.7,
            max_tokens=2000 + (num_samples * 300) # Adjust token based on expected output length
        )
        content = response.choices[0].message.content
        
        paths = re.findall(r"DIAGNOSTIC PATH #\d+:.*?(?=DIAGNOSTIC PATH #\d+:|$)", content, re.DOTALL)
        cleaned_paths = [path.strip() for path in paths if path.strip() and len(path.strip()) > 50] # Basic quality filter
        
        # Fallback strategies if not enough paths are generated
        if len(cleaned_paths) < num_samples:
            alt_paths = re.findall(r"(?:Path|Approach) \d+:.*?(?=(?:Path|Approach) \d+:|$)", content, re.DOTALL)
            cleaned_paths.extend([p.strip() for p in alt_paths if p.strip() and len(p.strip()) > 50 and p.strip() not in cleaned_paths])
        
        if len(cleaned_paths) < num_samples:
            paragraphs = content.split('\n\n')
            meaningful_paras = [p.strip() for p in paragraphs if len(p.strip()) > 100 and any(term in p.lower() for term in ['diagnosis', 'symptom', 'treatment', 'test', 'condition'])]
            cleaned_paths.extend([p for p in meaningful_paras if p not in cleaned_paths])
        
        # If still not enough, generate generic fallbacks (less ideal)
        while len(cleaned_paths) < num_samples:
            # Only add fallbacks if no paths were generated at all, otherwise return what was found.
            if cleaned_paths and num_samples > 1 : break # Avoid adding fallbacks if some good paths exist
            fallback_content = f"""DIAGNOSTIC PATH #{len(cleaned_paths)+1} (Fallback):
Based on symptoms: {input_text}
Consider differential diagnoses. Systematic evaluation needed.
Integrate medical knowledge: {medical_knowledge[:100]}..."""
            cleaned_paths.append(fallback_content)
            
        return cleaned_paths[:num_samples]
        
    except Exception as e:
        print(f"Error in async thought_generator_sampling: {e}")
        return [f"DIAGNOSTIC PATH #{i+1}: (Error Fallback) - {str(e)}" for i in range(num_samples)]

async def thought_generator_sequential(async_client_param: AsyncOpenAI, input_text: str, medical_knowledge: str, num_samples: int = 3) -> List[str]:
    """Enhanced sequential medical thought generator (ASYNC + Optimized Context)"""
    thoughts: List[str] = []
    explored_concepts: set[str] = set()
    
    base_context = f"Patient case: {input_text}\nMedical knowledge (summary): {medical_knowledge[:1000]}..."
    
    clinical_focus_areas = [
        "primary_differential_diagnosis", "secondary_conditions_comorbidities", 
        "diagnostic_testing_strategy", "treatment_planning_approach"
    ]
    
    for i in range(num_samples):
        current_focus = clinical_focus_areas[i % len(clinical_focus_areas)]
        
        exclusion_prompt = f"\nPreviously explored key concepts (avoid excessive overlap): {', '.join(list(explored_concepts)[:5])}" if explored_concepts else ""
        
        previous_context_summary = ""
        if thoughts:
            last_thought_summary = thoughts[-1][:250] + "..." if len(thoughts[-1]) > 250 else thoughts[-1] # Heuristic summary
            previous_context_summary = f"\nSummary of immediately preceding thought ({len(thoughts)}): {last_thought_summary}"
        
        messages: List[Dict[str, str]] = [ # Type hint for clarity
            {"role": "system", "content": f"""You are a senior medical specialist. Focus on {current_focus}. Provide specific medical terminology, diagnostic criteria, and reasoning. Path must start with 'DIAGNOSTIC PATH #{i+1} (Focus: {current_focus}):'."""},
            {"role": "user", "content": f"""{base_context}{previous_context_summary}{exclusion_prompt}
Generate a NEW, distinct diagnostic reasoning path focused on {current_focus} for the patient case.
Path should be self-contained. Include:
1. Specific medical terminology and diagnostic criteria.
2. Reference medical knowledge appropriately.
3. Detailed clinical reasoning.
4. Confidence levels for diagnostic hypotheses if natural.
Start with 'DIAGNOSTIC PATH #{i+1} (Focus: {current_focus}):'"""}
        ]
        
        try:
            response = await async_client_param.chat.completions.create(
                model="gpt-4o-mini", messages=messages, temperature=0.3, max_tokens=1500
            )
            thought_content = response.choices[0].message.content.strip()
            thoughts.append(thought_content)
            
            medical_terms_found = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:syndrome|disease|disorder|infection|cancer|tumor|condition|diagnosis))?)\b', thought_content)
            explored_concepts.update([term for term in medical_terms_found[:3] if len(term) > 3]) # Track first few major concepts
            
        except Exception as e:
            print(f"Error in async sequential thought generation (sample {i+1}): {e}")
            thoughts.append(f"DIAGNOSTIC PATH #{i+1} (Error Fallback): Focus: {current_focus} - Error: {str(e)}")
    
    return thoughts

async def _evaluate_single_thought_comprehensive(async_client_param: AsyncOpenAI, thought: str, input_text_summary: str, medical_knowledge_summary: str) -> Tuple[float, str]:
    """Helper to evaluate one thought comprehensively (ASYNC) - FOR INTERNAL USE"""
    evaluation_criteria = """
    Evaluate the diagnostic path based on medical standards (1-10 scale for EACH criterion):
    1. Clinical Reasoning Quality (Systematic approach, logical flow, medical accuracy)
    2. Diagnostic Accuracy (Appropriate differential diagnoses, evidence-based conclusions)
    3. Knowledge Integration (Effective use of medical knowledge from provided context)
    4. Completeness (Comprehensive assessment)
    5. Practical Applicability (Realistic clinical approach)
    """
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": f"""You are a senior medical expert evaluating diagnostic reasoning quality.
Use strict medical standards. {evaluation_criteria}
Your response MUST be formatted EXACTLY as follows:
SCORE: [Insert a single, overall numerical score from 1.0 to 10.0 here. Example: SCORE: 8.5]
JUSTIFICATION: [Detailed medical justification for this overall score, referencing the criteria. Explain WHY this score was given.]
"""},
        {"role": "user", "content": f"""Patient Case Summary: {input_text_summary}
Available Medical Knowledge Summary: {medical_knowledge_summary}

Evaluate this diagnostic path:
--- PATH START ---
{thought}
--- PATH END ---

Your response MUST start with "SCORE: " followed by the numerical score, then "JUSTIFICATION: ".
"""}
    ]
    try:
        response = await async_client_param.chat.completions.create(
            model="gpt-4o-mini", messages=messages, temperature=0.05, max_tokens=800
        )
        eval_content = response.choices[0].message.content

        score_match = re.search(r"SCORE:\s*([0-9]+\.?[0-9]*)", eval_content)
        score = 1.0 # Default low score
        justification_text = eval_content # Default justification is full content

        if score_match:
            try:
                potential_score = float(score_match.group(1))
                if 1.0 <= potential_score <= 10.0:
                    score = potential_score
                # Try to extract justification text after the score
                just_match = re.search(r"JUSTIFICATION:\s*(.*)", eval_content, re.DOTALL | re.IGNORECASE)
                if just_match:
                    justification_text = just_match.group(1).strip()
                else: # If specific justification not found, use content after score line
                    justification_text = eval_content[eval_content.find(score_match.group(0)) + len(score_match.group(0)):].strip()

            except ValueError:
                justification_text += "\n[INFO] Score value in 'SCORE: X.X' was not a valid float."
        else:
             justification_text += "\n[INFO] 'SCORE: X.X' line not found as expected."
             # Try a more general number search as a last resort for score
             fallback_num_match = re.search(r"([0-9]+\.?[0-9]+)", eval_content)
             if fallback_num_match:
                 try:
                     potential_fallback_score = float(fallback_num_match.group(1))
                     if 1.0 <= potential_fallback_score <= 10.0:
                         score = potential_fallback_score
                         justification_text += f"\n[INFO] Used fallback score extraction, found: {score}"
                 except ValueError:
                     pass # Stick to 1.0

        return score, justification_text
    except Exception as e:
        print(f"Error in _evaluate_single_thought_comprehensive: {e}")
        return 1.0, f"Evaluation failed: {str(e)} - assigned conservative score 1.0"

async def evaluate_comprehensive(async_client_param: AsyncOpenAI, thoughts: List[str], input_text: str, medical_knowledge: str) -> Tuple[List[float], List[str]]:
    if not thoughts:
        return [], []
    
    input_text_summary = input_text[:500] + "..." if len(input_text) > 500 else input_text
    medical_knowledge_summary = medical_knowledge[:1000] + "..." if len(medical_knowledge) > 1000 else medical_knowledge

    tasks = [_evaluate_single_thought_comprehensive(async_client_param, thought, input_text_summary, medical_knowledge_summary) for thought in thoughts]
    results = await asyncio.gather(*tasks)
    
    scores_list = [result[0] for result in results]
    justifications_list = [result[1] for result in results]
    
    return scores_list, justifications_list

async def evaluate_comparative_detailed(async_client_param: AsyncOpenAI, thoughts: List[str], input_text: str, medical_knowledge: str) -> Tuple[List[float], str]:
    if not thoughts:
        return [], "No thoughts to compare."
    
    input_text_summary = input_text[:500] + "..." if len(input_text) > 500 else input_text
    medical_knowledge_summary = medical_knowledge[:1000] + "..." if len(medical_knowledge) > 1000 else medical_knowledge
    full_eval_content = "Comparative evaluation initiated."

    try:
        comparison_text = "\n\n" + "="*30 + "\n\n".join([f"DIAGNOSTIC PATH {i+1}:\n{thought}" for i, thought in enumerate(thoughts)])
        
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": """You are a senior attending physician. Compare the provided diagnostic approaches.
Rank paths based on medical accuracy, clinical reasoning, knowledge integration, and applicability.
Your response MUST include a ranking list formatted EXACTLY as:
RANKING:
PATH 1: Score=X.X
PATH N: Score=Y.Y

Then, provide a JUSTIFICATION section detailing reasons for scores and ranking."""},
            {"role": "user", "content": f"""Patient Case Summary: {input_text_summary}
Medical Knowledge Summary: {medical_knowledge_summary}

Diagnostic Paths to Compare:
{comparison_text}

Provide your comparative analysis. Start with the "RANKING:" list, then "JUSTIFICATION:"."""}
        ]
        
        response = await async_client_param.chat.completions.create(
            model="gpt-4o-mini", messages=messages, temperature=0.05, max_tokens=2500
        )
        full_eval_content = response.choices[0].message.content
        
        scores_list = [1.0] * len(thoughts) # Default low score
        
        ranking_section_match = re.search(r"RANKING:(.*?)JUSTIFICATION:", full_eval_content, re.DOTALL | re.IGNORECASE)
        if not ranking_section_match: # Try finding RANKING section alone if JUSTIFICATION is missing
             ranking_section_match = re.search(r"RANKING:(.*)", full_eval_content, re.DOTALL | re.IGNORECASE)

        if ranking_section_match:
            ranking_text = ranking_section_match.group(1)
            path_score_pattern = r'PATH\s*(\d+):\s*Score=([0-9]+\.?[0-9]*)'
            matches = re.findall(path_score_pattern, ranking_text)
            
            parsed_scores_count = 0
            for path_num_str, score_str in matches:
                try:
                    path_idx = int(path_num_str) - 1
                    score_val = float(score_str)
                    if 0 <= path_idx < len(thoughts) and 1.0 <= score_val <= 10.0:
                        scores_list[path_idx] = score_val
                        parsed_scores_count +=1
                except (ValueError, IndexError):
                    print(f"Warning: Could not parse path score during comparative eval: Path {path_num_str}, Score {score_str}")
            if parsed_scores_count == 0:
                 full_eval_content += "\n[INFO] No scores parsed from RANKING section using PATH X: Score=Y.Z format."
        else:
            full_eval_content += "\n[INFO] Could not find RANKING section for detailed score parsing in comparative eval."
            # Minimal fallback: if only one thought, try to find any score for it
            if len(thoughts) == 1:
                score_match = re.search(r"Score=([0-9]+\.?[0-9]*)", full_eval_content)
                if score_match:
                    try: scores_list[0] = float(score_match.group(1))
                    except: pass


        return scores_list, full_eval_content
        
    except Exception as e:
        print(f"Error in async comparative evaluation: {e}")
        return [1.0 + random.uniform(0, 0.5) for _ in thoughts], f"Comparative evaluation failed: {str(e)}"

# async def tot_breadth_first_search(async_client_param: AsyncOpenAI, input_text: str, medical_knowledge: str, 
#                                    thought_generator_func: Callable, evaluator_func: Callable,
#                                    step_limit: int = 2, breadth: int = 2, samples_per_step: int = 2) -> Tuple[List[str], List[List[float]]]:
#     print(f"Starting Async Medical BFS: Steps={step_limit}, Breadth={breadth}, Samples/Step={samples_per_step}")
    
#     current_states: List[Dict[str, Any]] = [{
#         "context": f"Initial state for: {input_text[:100]}...", "path": [], "score": 5.0,
#         "confidence": 0.5, "focus_areas": ["initial_assessment"], "medical_concepts": set()
#     }]
#     all_step_scores_history: List[List[float]] = []
    
#     for step_num in range(step_limit):
#         print(f"  BFS Step {step_num+1}/{step_limit}: Processing {len(current_states)} states...")
#         if not current_states: break

#         next_candidate_states: List[Dict[str, Any]] = []
        
#         generation_tasks_for_step = []
#         state_indices_for_tasks = [] # To map generated thoughts back to their parent state's concepts

#         for state_idx, state_data in enumerate(current_states):
#             medical_focus_progression = ["differential_diagnosis", "diagnostic_testing", "treatment_planning", "prognosis_monitoring"]
#             current_focus_for_state = medical_focus_progression[step_num % len(medical_focus_progression)]
            
#             # Context for thought generation
#             prev_step_summary = state_data['path'][-1][:200] + "..." if state_data['path'] else "N/A"
#             context_for_gen = f"Patient Case: {input_text}\nPrevious Step Summary: {prev_step_summary}\nCurrent Focus: {current_focus_for_state}"
            
#             generation_tasks_for_step.append(
#                 thought_generator_func(async_client_param, context_for_gen, medical_knowledge, samples_per_step, current_focus_for_state)
#             )
#             state_indices_for_tasks.append(state_idx) # Store index to access parent state_data later

#         # Gather all generated thoughts for this step
#         list_of_new_thoughts_per_state = await asyncio.gather(*generation_tasks_for_step)

#         # Process generated thoughts and create next_candidate_states
#         for i, new_thoughts_group in enumerate(list_of_new_thoughts_per_state):
#             parent_state_data = current_states[state_indices_for_tasks[i]] # Get original parent state
#             current_focus_for_state = medical_focus_progression[step_num % len(medical_focus_progression)] # Re-fetch focus

#             for thought_text in new_thoughts_group:
#                 if thought_text and len(thought_text.strip()) > 30: # Quality filter
#                     new_path_nodes = parent_state_data["path"] + [thought_text.strip()]
#                     found_medical_terms = set(re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:syndrome|disease|disorder|infection|condition))?)\b', thought_text))
#                     next_candidate_states.append({
#                         "context": thought_text, "path": new_path_nodes, "score": 0.0, "confidence": 0.0, # Score to be filled
#                         "focus_areas": parent_state_data["focus_areas"] + [current_focus_for_state],
#                         "medical_concepts": parent_state_data["medical_concepts"].union(found_medical_terms)
#                     })
        
#         if not next_candidate_states:
#             print(f"  BFS Step {step_num+1}: No valid candidate states generated.")
#             break 

#         # Evaluate all candidate thoughts for this step
#         thought_texts_for_eval = [s["context"] for s in next_candidate_states]
#         eval_scores_for_step, _ = await evaluator_func(async_client_param, thought_texts_for_eval, input_text, medical_knowledge)
        
#         if not isinstance(eval_scores_for_step, list) or len(eval_scores_for_step) != len(next_candidate_states):
#             print(f"  Warning: Score list length mismatch in BFS Step {step_num+1}. Defaulting scores.")
#             eval_scores_for_step = [1.0] * len(next_candidate_states)

#         # Update candidate states with scores
#         for i_state, score_value in enumerate(eval_scores_for_step):
#             validated_score = max(1.0, min(10.0, float(score_value if score_value is not None else 1.0)))
#             next_candidate_states[i_state]["score"] = validated_score
#             next_candidate_states[i_state]["confidence"] = min(1.0, validated_score / 10.0)
        
#         all_step_scores_history.append([s['score'] for s in next_candidate_states]) # Save scores of all candidates this step
        
#         # Sort candidates by score and diversity (number of unique medical concepts)
#         next_candidate_states.sort(key=lambda x: (x["score"], len(x["medical_concepts"])), reverse=True)
        
#         # Select next states for current_states based on breadth and diversity
#         selected_next_states: List[Dict[str, Any]] = []
#         concepts_covered_in_selection = set()
#         for candidate_state in next_candidate_states:
#             if len(selected_next_states) >= breadth:
#                 break
#             # Prefer states that introduce new concepts or are high scoring
#             if (candidate_state["medical_concepts"] - concepts_covered_in_selection) or not selected_next_states or candidate_state["score"] > 7.0:
#                 selected_next_states.append(candidate_state)
#                 concepts_covered_in_selection.update(candidate_state["medical_concepts"])
        
#         # If diversity filter was too strict, fill remaining breadth with best score
#         idx = 0
#         while len(selected_next_states) < breadth and idx < len(next_candidate_states):
#             if next_candidate_states[idx] not in selected_next_states: # Avoid duplicates
#                 selected_next_states.append(next_candidate_states[idx])
#             idx +=1
            
#         current_states = selected_next_states
#         if not current_states:
#             print(f"  BFS Step {step_num+1}: No states selected after evaluation and pruning.")
#             break
#         # print(f"  BFS Step {step_num+1} - Best scores selected: {[f'{s["score"]:.1f}' for s in current_states]}")
#         # Corrected line:
#         # print(f"  BFS Step {step_num+1} - Best scores selected: {[f'{s['score']:.1f}' for s in current_states]}")
#         # Corrected code:
#         best_scores_str = ", ".join([f"{s['score']:.1f}" for s in current_states])
#         print(f"  BFS Step {step_num+1} - Best scores selected: [{best_scores_str}]")    
#     if current_states: # If any states remain after all steps
#         best_overall_state = max(current_states, key=lambda x: x["score"])
#         return best_overall_state["path"], all_step_scores_history
#     return [], all_step_scores_history
async def tot_breadth_first_search(async_client_param: AsyncOpenAI, input_text: str, medical_knowledge: str, 
                                   thought_generator_func: Callable, evaluator_func: Callable,
                                   step_limit: int = 2, breadth: int = 2, samples_per_step: int = 2) -> Tuple[List[str], List[List[float]]]:
    print(f"Starting Async Medical BFS: Steps={step_limit}, Breadth={breadth}, Samples/Step={samples_per_step}")
    
    current_states: List[Dict[str, Any]] = [{
        "context": f"Initial state for: {input_text[:100]}...", "path": [], "score": 5.0,
        "confidence": 0.5, "focus_areas": ["initial_assessment"], "medical_concepts": set()
    }]
    all_step_scores_history: List[List[float]] = []
    
    for step_num in range(step_limit):
        print(f"  BFS Step {step_num+1}/{step_limit}: Processing {len(current_states)} states...")
        if not current_states: break

        next_candidate_states: List[Dict[str, Any]] = []
        
        generation_tasks_for_step = []
        state_indices_for_tasks = [] 

        for state_idx, state_data in enumerate(current_states):
            medical_focus_progression = ["differential_diagnosis", "diagnostic_testing", "treatment_planning", "prognosis_monitoring"]
            current_focus_for_state = medical_focus_progression[step_num % len(medical_focus_progression)]
            
            prev_step_summary = state_data['path'][-1][:200] + "..." if state_data['path'] else "N/A"
            context_for_gen = f"Patient Case: {input_text}\nPrevious Step Summary: {prev_step_summary}\nCurrent Focus: {current_focus_for_state}"
            
            # Conditional argument passing for thought generator
            if thought_generator_func.__name__ == "thought_generator_sampling":
                generation_tasks_for_step.append(
                    thought_generator_func(async_client_param, context_for_gen, medical_knowledge, samples_per_step, current_focus_for_state)
                )
            else: # For thought_generator_sequential or others not needing focus_area
                generation_tasks_for_step.append(
                    thought_generator_func(async_client_param, context_for_gen, medical_knowledge, samples_per_step)
                )
            state_indices_for_tasks.append(state_idx)

        list_of_new_thoughts_per_state = await asyncio.gather(*generation_tasks_for_step)

        for i, new_thoughts_group in enumerate(list_of_new_thoughts_per_state):
            parent_state_data = current_states[state_indices_for_tasks[i]] 
            current_focus_for_state = medical_focus_progression[step_num % len(medical_focus_progression)] 

            for thought_text in new_thoughts_group:
                if thought_text and len(thought_text.strip()) > 30: 
                    new_path_nodes = parent_state_data["path"] + [thought_text.strip()]
                    found_medical_terms = set(re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:syndrome|disease|disorder|infection|condition))?)\b', thought_text))
                    next_candidate_states.append({
                        "context": thought_text, "path": new_path_nodes, "score": 0.0, "confidence": 0.0, 
                        "focus_areas": parent_state_data["focus_areas"] + [current_focus_for_state],
                        "medical_concepts": parent_state_data["medical_concepts"].union(found_medical_terms)
                    })
        
        if not next_candidate_states:
            print(f"  BFS Step {step_num+1}: No valid candidate states generated.")
            break 

        thought_texts_for_eval = [s["context"] for s in next_candidate_states]
        eval_scores_for_step, _ = await evaluator_func(async_client_param, thought_texts_for_eval, input_text, medical_knowledge)
        
        if not isinstance(eval_scores_for_step, list) or len(eval_scores_for_step) != len(next_candidate_states):
            print(f"  Warning: Score list length mismatch in BFS Step {step_num+1}. Defaulting scores.")
            eval_scores_for_step = [1.0] * len(next_candidate_states)

        for i_state, score_value in enumerate(eval_scores_for_step):
            validated_score = max(1.0, min(10.0, float(score_value if score_value is not None else 1.0)))
            next_candidate_states[i_state]["score"] = validated_score
            next_candidate_states[i_state]["confidence"] = min(1.0, validated_score / 10.0)
        
        all_step_scores_history.append([s['score'] for s in next_candidate_states]) 
        
        next_candidate_states.sort(key=lambda x: (x["score"], len(x["medical_concepts"])), reverse=True)
        
        selected_next_states: List[Dict[str, Any]] = []
        concepts_covered_in_selection = set()
        for candidate_state in next_candidate_states:
            if len(selected_next_states) >= breadth:
                break
            if (candidate_state["medical_concepts"] - concepts_covered_in_selection) or not selected_next_states or candidate_state["score"] > 7.0:
                selected_next_states.append(candidate_state)
                concepts_covered_in_selection.update(candidate_state["medical_concepts"])
        
        idx = 0
        while len(selected_next_states) < breadth and idx < len(next_candidate_states):
            if next_candidate_states[idx] not in selected_next_states: 
                selected_next_states.append(next_candidate_states[idx])
            idx +=1
            
        current_states = selected_next_states
        if not current_states:
            print(f"  BFS Step {step_num+1}: No states selected after evaluation and pruning.")
            break
        
        best_scores_str = ", ".join([f"{s['score']:.1f}" for s in current_states])
        print(f"  BFS Step {step_num+1} - Best scores selected: [{best_scores_str}]")    
    
    if current_states: 
        best_overall_state = max(current_states, key=lambda x: x["score"])
        return best_overall_state["path"], all_step_scores_history
    return [], all_step_scores_history

# async def tot_depth_first_search(async_client_param: AsyncOpenAI, input_text: str, medical_knowledge: str, 
#                                  thought_generator_func: Callable, evaluator_func: Callable,
#                                  step_limit: int = 2, threshold: float = 5.5, 
#                                  samples_per_step: int = 2, max_paths_to_find: int = 1) -> Tuple[List[str], List[List[float]]]:
#     print(f"Starting Async Medical DFS: Threshold={threshold}, Samples/Step={samples_per_step}, MaxPaths={max_paths_to_find}")
    
#     final_paths_collection: List[Tuple[List[str], float, List[float]]] = []
#     # Use a dictionary for mutable counter in recursive scope
#     exploration_tracker = {'count': 0, 'limit': max_paths_to_find * step_limit * samples_per_step + 10} # Heuristic limit

#     async def dfs_recursive_inner_logic(current_thought_context_summary: str, depth: int, 
#                                         current_path_accumulated: List[str], current_path_scores: List[float], 
#                                         path_explored_concepts: set):
        
#         if exploration_tracker['count'] >= exploration_tracker['limit'] or \
#            len(final_paths_collection) >= max_paths_to_find:
#             return

#         exploration_tracker['count'] += 1
        
#         if depth >= step_limit: # Path completed
#             if current_path_scores:
#                 avg_path_score = sum(current_path_scores) / len(current_path_scores)
#                 if avg_path_score >= threshold:
#                     final_paths_collection.append((list(current_path_accumulated), avg_path_score, list(current_path_scores)))
#                     print(f"  DFS: Found valid path (AvgScore={avg_path_score:.2f}, Depth={depth})")
#             return

#         medical_focus_areas = ["symptom_analysis", "differential_diagnosis", "diagnostic_testing", "treatment_planning"]
#         current_focus_for_depth = medical_focus_areas[depth % len(medical_focus_areas)]
        
#         context_for_gen = (f"Patient Case Summary: {input_text[:300]}...\n"
#                            f"Medical Knowledge Snippet: {medical_knowledge[:300]}...\n"
#                            f"Current Path Focus: {current_focus_for_depth}, Depth: {depth+1}/{step_limit}\n"
#                            f"Last thought in path (summary): {current_thought_context_summary[:200]}...\n"
#                            f"Explored concepts in this path (avoid major overlap): {', '.join(list(path_explored_concepts)[:3]) if path_explored_concepts else 'None'}")

#         new_candidate_thoughts = await thought_generator_func(async_client_param, context_for_gen, medical_knowledge, samples_per_step, current_focus_for_depth)
        
#         quality_filtered_thoughts = [t for t in new_candidate_thoughts if t and len(t.strip()) > 30]
#         if not quality_filtered_thoughts: return

#         eval_scores_for_thoughts, _ = await evaluator_func(async_client_param, quality_filtered_thoughts, input_text, medical_knowledge)
        
#         if not isinstance(eval_scores_for_thoughts, list) or len(eval_scores_for_thoughts) != len(quality_filtered_thoughts):
#             eval_scores_for_thoughts = [1.0] * len(quality_filtered_thoughts) # Fallback

#         # Sort thoughts by score to explore promising ones first
#         scored_candidate_thoughts = sorted(zip(quality_filtered_thoughts, eval_scores_for_thoughts), key=lambda x: x[1], reverse=True)
        
#         for thought_node_text, thought_score in scored_candidate_thoughts:
#             if len(final_paths_collection) >= max_paths_to_find: break # Stop if enough paths found globally

#             # Pruning: if score is too low, don't explore further. Less strict for initial steps.
#             if thought_score < threshold and (depth > 0 or thought_score < 4.0): 
#                 continue

#             current_path_accumulated.append(thought_node_text.strip())
#             current_path_scores.append(thought_score)
            
#             newly_added_medical_concepts = set(re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:syndrome|disease|disorder|infection|condition))?)\b', thought_node_text))
            
#             await dfs_recursive_inner_logic(thought_node_text, depth + 1, current_path_accumulated, 
#                                             current_path_scores, path_explored_concepts.union(newly_added_medical_concepts))
            
#             current_path_accumulated.pop() # Backtrack path
#             current_path_scores.pop()      # Backtrack scores

#     # Initial call to the recursive helper
#     await dfs_recursive_inner_logic(input_text, 0, [], [], set())
    
#     if final_paths_collection:
#         final_paths_collection.sort(key=lambda x: x[1], reverse=True) # Sort by average score
#         best_path_nodes, best_avg_score, best_path_step_scores = final_paths_collection[0]
#         print(f"DFS found {len(final_paths_collection)} valid paths. Best avg_score: {best_avg_score:.2f}")
#         return best_path_nodes, [best_path_step_scores] # Return nodes and scores of the single best path
    
#     print("DFS found no paths meeting threshold criteria after extensive exploration.")
#     return [], []
async def tot_depth_first_search(async_client_param: AsyncOpenAI, input_text: str, medical_knowledge: str, 
                                 thought_generator_func: Callable, evaluator_func: Callable,
                                 step_limit: int = 2, threshold: float = 5.5, 
                                 samples_per_step: int = 2, max_paths_to_find: int = 1) -> Tuple[List[str], List[List[float]]]:
    print(f"Starting Async Medical DFS: Threshold={threshold}, Samples/Step={samples_per_step}, MaxPaths={max_paths_to_find}")
    
    final_paths_collection: List[Tuple[List[str], float, List[float]]] = []
    exploration_tracker = {'count': 0, 'limit': max_paths_to_find * step_limit * samples_per_step + 15} # Adjusted limit

    async def dfs_recursive_inner_logic(current_thought_context_summary: str, depth: int, 
                                        current_path_accumulated: List[str], current_path_scores: List[float], 
                                        path_explored_concepts: set):
        
        if exploration_tracker['count'] >= exploration_tracker['limit'] or \
           len(final_paths_collection) >= max_paths_to_find:
            return

        exploration_tracker['count'] += 1
        
        if depth >= step_limit: 
            if current_path_scores: # Ensure there are scores to average
                avg_path_score = sum(current_path_scores) / len(current_path_scores) if len(current_path_scores) > 0 else 0
                if avg_path_score >= threshold:
                    final_paths_collection.append((list(current_path_accumulated), avg_path_score, list(current_path_scores)))
                    print(f"  DFS: Found valid path (AvgScore={avg_path_score:.2f}, Depth={depth})")
            return

        medical_focus_areas = ["symptom_analysis", "differential_diagnosis", "diagnostic_testing", "treatment_planning"]
        current_focus_for_depth = medical_focus_areas[depth % len(medical_focus_areas)]
        
        context_for_gen = (f"Patient Case Summary: {input_text[:300]}...\n"
                           f"Medical Knowledge Snippet: {medical_knowledge[:300]}...\n"
                           f"Current Path Focus: {current_focus_for_depth}, Depth: {depth+1}/{step_limit}\n"
                           f"Last thought in path (summary): {current_thought_context_summary[:200]}...\n"
                           f"Explored concepts in this path (avoid major overlap): {', '.join(list(path_explored_concepts)[:3]) if path_explored_concepts else 'None'}")

        # Conditional argument passing for thought generator
        if thought_generator_func.__name__ == "thought_generator_sampling":
            new_candidate_thoughts = await thought_generator_func(async_client_param, context_for_gen, medical_knowledge, samples_per_step, current_focus_for_depth)
        else: # For thought_generator_sequential
            new_candidate_thoughts = await thought_generator_func(async_client_param, context_for_gen, medical_knowledge, samples_per_step)

        quality_filtered_thoughts = [t for t in new_candidate_thoughts if t and len(t.strip()) > 30]
        if not quality_filtered_thoughts: return

        eval_scores_for_thoughts, _ = await evaluator_func(async_client_param, quality_filtered_thoughts, input_text, medical_knowledge)
        
        if not isinstance(eval_scores_for_thoughts, list) or len(eval_scores_for_thoughts) != len(quality_filtered_thoughts):
            eval_scores_for_thoughts = [1.0] * len(quality_filtered_thoughts) 

        scored_candidate_thoughts = sorted(zip(quality_filtered_thoughts, eval_scores_for_thoughts), key=lambda x: x[1], reverse=True)
        
        for thought_node_text, thought_score in scored_candidate_thoughts:
            if len(final_paths_collection) >= max_paths_to_find: break 

            if thought_score < threshold and (depth > 0 or thought_score < 4.0): 
                continue

            current_path_accumulated.append(thought_node_text.strip())
            current_path_scores.append(thought_score)
            
            newly_added_medical_concepts = set(re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:syndrome|disease|disorder|infection|condition))?)\b', thought_node_text))
            
            await dfs_recursive_inner_logic(thought_node_text, depth + 1, current_path_accumulated, 
                                            current_path_scores, path_explored_concepts.union(newly_added_medical_concepts))
            
            current_path_accumulated.pop() 
            current_path_scores.pop()      

    await dfs_recursive_inner_logic(input_text, 0, [], [], set())
    
    if final_paths_collection:
        final_paths_collection.sort(key=lambda x: x[1], reverse=True) 
        best_path_nodes, best_avg_score, best_path_step_scores = final_paths_collection[0]
        print(f"DFS found {len(final_paths_collection)} valid paths. Best avg_score: {best_avg_score:.2f}")
        return best_path_nodes, [best_path_step_scores] 
    
    print("DFS found no paths meeting threshold criteria after extensive exploration.")
    return [], []

# --- ToTResult Class and Helper Functions (No async changes needed here) ---
class ToTResult:
    def __init__(self, method_name: str, reasoning_paths: List[str], final_answer: str, 
                 execution_time: float, confidence_score: float, step_scores: List[Any], 
                 total_api_calls: int, quality_metrics: Optional[Dict[str, Any]] = None):
        self.method_name = method_name
        self.reasoning_paths = reasoning_paths if reasoning_paths else [] # Ensure it's a list
        self.final_answer = final_answer
        self.execution_time = execution_time
        self.confidence_score = confidence_score
        self.step_scores = step_scores if step_scores else [] # Ensure it's a list
        self.total_api_calls = total_api_calls # Note: Currently not accurately tracked in this refactor
        self.quality_metrics = quality_metrics or {}
        self._compute_derived_metrics()
    
    def _compute_derived_metrics(self):
        if self.reasoning_paths: # Check if list is not empty
            path_lengths = [len(path.split()) for path in self.reasoning_paths if path] # ensure path is not empty
            if path_lengths: # ensure path_lengths is not empty before np.mean/std
                 self.quality_metrics['avg_path_length'] = np.mean(path_lengths)
                 self.quality_metrics['path_length_std'] = np.std(path_lengths)
            else: # Handle case where all paths are empty or reasoning_paths was empty
                 self.quality_metrics['avg_path_length'] = 0
                 self.quality_metrics['path_length_std'] = 0

            self.quality_metrics['total_paths'] = len(self.reasoning_paths)
            # Path diversity needs at least one path
            if self.reasoning_paths:
                 self.quality_metrics['path_diversity'] = len(set(path[:100] for path in self.reasoning_paths if path)) / len(self.reasoning_paths) if len(self.reasoning_paths) > 0 else 0
            else:
                 self.quality_metrics['path_diversity'] = 0
        
        flat_scores = []
        for step_score_list in self.step_scores: # step_scores is List[List[float]] for BFS
            if isinstance(step_score_list, list):
                flat_scores.extend(s for s in step_score_list if isinstance(s, (int, float)))
            elif isinstance(step_score_list, (int, float)): # DFS might return List[float] directly for the best path
                flat_scores.append(step_score_list)

        if flat_scores:
            self.quality_metrics.update({
                'score_mean': np.mean(flat_scores), 'score_std': np.std(flat_scores),
                'score_min': min(flat_scores), 'score_max': max(flat_scores),
                'score_stability': 1.0 - (np.std(flat_scores) / max(np.mean(flat_scores), 1e-6)) # Avoid div by zero
            })
    
    def get_overall_quality_score(self) -> float:
        base_confidence = self.confidence_score
        path_quality_bonus = self.quality_metrics.get('path_diversity', 0.0) * 0.1
        stability_bonus = self.quality_metrics.get('score_stability', 0.0) * 0.1
        efficiency_bonus = 0.0 # API calls not tracked, so efficiency bonus is hard to calculate here
        overall_score = base_confidence + path_quality_bonus + stability_bonus + efficiency_bonus
        return min(1.0, max(0.0, overall_score)) # Clamp between 0 and 1
    
    # def get_summary_dict(self) -> Dict[str, Any]:
    #     return {
    #         'method': self.method_name, 'confidence': self.confidence_score,
    #         'overall_quality': self.get_overall_quality_score(),
    #         'execution_time': self.execution_time, 'num_paths': len(self.reasoning_paths),
    #         'api_calls': self.total_api_calls, **self.quality_metrics
    #     }
    def get_summary_dict(self) -> Dict[str, Any]:
        return {
            'method': self.method_name, 
            'confidence_score': self.confidence_score, # Changed key from 'confidence'
            'overall_quality': self.get_overall_quality_score(),
            'execution_time': self.execution_time, 
            'num_paths': len(self.reasoning_paths), 
            'api_calls': self.total_api_calls, 
            **self.quality_metrics 
        }

def extract_answer_from_paths(reasoning_paths: List[str], input_text: str, medical_knowledge: str) -> str:
    if not reasoning_paths:
        return "Unable to generate diagnosis: No reasoning paths provided."
    
    # Simple approach: use the first (often best) path or a summary of all for synthesis
    # For a more robust approach, use the `final_answer` part of a ToTResult if available from a structured LLM call.
    # This function is a simplified synthesizer if only paths are given.
    
    # Let's assume reasoning_paths contains the textual reasoning.
    # We'll use the first path as the primary evidence for this simplified extraction.
    primary_reasoning = reasoning_paths[0] if reasoning_paths else "No specific reasoning path available."
    
    # This could be an LLM call to synthesize, but for now, a template:
    answer_template = f"""Based on the analysis of the patient's case and medical knowledge:
Patient Input Summary: {input_text[:300]}...
Key Reasoning Considered:
{primary_reasoning[:500]}... {"(truncated)" if len(primary_reasoning)>500 else ""}

A comprehensive medical assessment suggests further clinical correlation.
The diagnostic possibilities based on the reasoning include [Placeholder for Diagnosis].
Recommended tests may involve [Placeholder for Tests].
Potential treatment approaches could include [Placeholder for Treatment].
(Note: This is a summarized extraction. Detailed final answer usually comes from a dedicated synthesis step in ToT.)
"""
    # A proper implementation would involve an LLM call here with all paths for synthesis.
    # The placeholder values should be extracted or inferred by an LLM.
    # For now, it just indicates what should be there.
    # The ToTResult.final_answer from the benchmark is usually more complete.
    return answer_template # This is a very basic extraction

def calculate_confidence_score(reasoning_paths: List[str], step_scores: List[Any], method_name: Optional[str] = None) -> float:
    if not step_scores or not reasoning_paths: return 0.1
    
    all_scores_flat: List[float] = []
    for step_s_list in step_scores: # step_scores is List[List[float]] for BFS, List[float] for DFS best path
        if isinstance(step_s_list, list):
            all_scores_flat.extend(s for s in step_s_list if isinstance(s, (int, float)))
        elif isinstance(step_s_list, (int, float)): # For DFS best path scores
            all_scores_flat.append(step_s_list)
            
    if not all_scores_flat: return 0.2
    
    avg_score_val = np.mean(all_scores_flat)
    base_confidence_val = min(1.0, max(0.0, (avg_score_val - 1.0) / 9.0)) # Scale 1-10 to 0-1
    
    path_quality_factors_list: List[float] = []
    for path_text in reasoning_paths:
        if not path_text or len(path_text.strip()) < 20: path_quality_factors_list.append(0.1); continue
        medical_terms_count = len(re.findall(r'\b(diagnosis|symptom|treatment|clinical|patient|condition)\b', path_text.lower(), re.IGNORECASE))
        medical_density_val = medical_terms_count / max(len(path_text.split()), 1)
        path_quality_factors_list.append(min(1.0, medical_density_val * 2.0)) # Simple quality proxy
    
    avg_path_quality_val = np.mean(path_quality_factors_list) if path_quality_factors_list else 0.3
    
    score_consistency_val = 1.0 - (np.std(all_scores_flat) / max(avg_score_val, 1e-6)) if len(all_scores_flat) > 1 else 1.0
    
    method_bonus_val = 0.05 if method_name and ('BFS' in method_name or 'Comparative' in method_name) else 0.0
    
    final_confidence_score = (base_confidence_val * 0.5 + avg_path_quality_val * 0.3 + score_consistency_val * 0.15 + method_bonus_val)
    return round(min(0.98, max(0.12, final_confidence_score + random.uniform(-0.02, 0.02))), 3)


# async def run_tot_benchmark(async_client_param: AsyncOpenAI, input_text: str, medical_knowledge: str, 
#                             test_methods_list: Optional[List[str]] = None) -> Dict[str, ToTResult]:
#     if test_methods_list is None: # Default methods if none provided
#         test_methods_list = [
#             "Sampling_Comparative_BFS", "Sequential_Comparative_BFS",
#             "Sampling_Comprehensive_BFS", "Sequential_Comparative_DFS",
#             "Sampling_Comprehensive_DFS" 
#         ]
    
#     # Async function maps
#     async_thought_generators: Dict[str, Callable] = {"Sampling": thought_generator_sampling, "Sequential": thought_generator_sequential}
#     async_evaluators: Dict[str, Callable] = {"Comprehensive": evaluate_comprehensive, "Comparative": evaluate_comparative_detailed}
#     async_search_algorithms: Dict[str, Callable] = {"BFS": tot_breadth_first_search, "DFS": tot_depth_first_search}
    
#     benchmark_results_dict: Dict[str, ToTResult] = {}
    
#     print(f"Starting Async Medical ToT Benchmark with {len(test_methods_list)} methods")
#     print("="*60)
    
#     for method_combo_str in test_methods_list:
#         gen_name, eval_name, search_name = method_combo_str.split("_")
#         print(f"\n--- Testing {method_combo_str} (Async) ---")
        
#         async_gen_func = async_thought_generators.get(gen_name)
#         async_eval_func = async_evaluators.get(eval_name)
#         async_search_func = async_search_algorithms.get(search_name)
        
#         if not all([async_gen_func, async_eval_func, async_search_func]):
#             print(f"Warning: Could not find async methods for {method_combo_str}")
#             benchmark_results_dict[method_combo_str] = ToTResult(method_combo_str, [], "Method component not found", 0.0, 0.05, [], 0, {'error': True})
#             continue
        
#         start_time_val = time.time()
#         reasoning_paths_list: List[str] = []
#         step_scores_list: List[Any] = []
        
#         try:
#             # Default parameters for search algorithms (can be tuned)
#             search_params = { # Parameters can be specific to method combination if needed
#                 "BFS": {"step_limit": 2, "breadth": 2, "samples_per_step": 2},
#                 "DFS": {"step_limit": 2, "threshold": 5.0, "samples_per_step": 2, "max_paths_to_find": 1}
#             }
#             current_search_params = search_params.get(search_name, {})

#             reasoning_paths_list, step_scores_list = await async_search_func(
#                 async_client_param, input_text, medical_knowledge, 
#                 async_gen_func, async_eval_func, **current_search_params
#             )
            
#             execution_time_val = time.time() - start_time_val
#             # Simplified answer extraction for benchmark summary
#             # A more sophisticated synthesis LLM call could be used here for higher quality 'final_answer'
#             final_answer_str = extract_answer_from_paths(reasoning_paths_list, input_text, medical_knowledge) 
#             if reasoning_paths_list: # If paths were found, use content from the best path for a slightly better answer
#                  final_answer_str = f"Primary Diagnosis Path suggests: {reasoning_paths_list[0][:800]}..."

#             confidence_score_val = calculate_confidence_score(reasoning_paths_list, step_scores_list, method_combo_str)
            
#             result_obj = ToTResult(
#                 method_combo_str, reasoning_paths_list, final_answer_str, execution_time_val, 
#                 confidence_score_val, step_scores_list, 0, # API calls not tracked
#                 {'generator_type': gen_name, 'evaluator_type': eval_name, 'search_type': search_name}
#             )
#             benchmark_results_dict[method_combo_str] = result_obj
            
#             print(f"✓ {method_combo_str} completed in {execution_time_val:.2f}s")
#             print(f"  Confidence: {confidence_score_val:.3f}, Overall Quality: {result_obj.get_overall_quality_score():.3f}, Paths: {len(reasoning_paths_list)}")
            
#         except Exception as e_inner:
#             exec_time_on_error = time.time() - start_time_val
#             print(f"✗ Error in {method_combo_str} after {exec_time_on_error:.2f}s: {str(e_inner)}")
#             benchmark_results_dict[method_combo_str] = ToTResult(
#                 method_combo_str, [], f"Error: {str(e_inner)}", exec_time_on_error, 0.05, [], 0, {'error': True}
#             )

#     print("\n" + "="*60 + "\nBENCHMARK SUMMARY (Async)\n" + "="*60)
#     if benchmark_results_dict:
#         sorted_results_list = sorted(
#             benchmark_results_dict.items(), 
#             key=lambda item: item[1].get_overall_quality_score(), 
#             reverse=True
#         )
#         for i, (method_str, res_obj) in enumerate(sorted_results_list, 1):
#             print(f"{i:2d}. {method_str}: Q={res_obj.get_overall_quality_score():.3f} "
#                   f"(Conf={res_obj.confidence_score:.3f}, Time={res_obj.execution_time:.1f}s, "
#                   f"Paths={len(res_obj.reasoning_paths)})")
#     return benchmark_results_dict
async def run_tot_benchmark(async_client_param: AsyncOpenAI, input_text: str, medical_knowledge: str, 
                            test_methods_list: Optional[List[str]] = None) -> Dict[str, ToTResult]:
    if test_methods_list is None: # Default methods if none provided
        test_methods_list = [
            "Sampling_Comprehensive_BFS", 
            "Sampling_Comprehensive_DFS", 
            "Sampling_Comparative_BFS", 
            "Sampling_Comparative_DFS", 
            "Sequential_Comprehensive_BFS", 
            "Sequential_Comprehensive_DFS", 
            "Sequential_Comparative_BFS", 
            "Sequential_Comparative_DFS"
        ]
    
    async_thought_generators: Dict[str, Callable] = {"Sampling": thought_generator_sampling, "Sequential": thought_generator_sequential}
    async_evaluators: Dict[str, Callable] = {"Comprehensive": evaluate_comprehensive, "Comparative": evaluate_comparative_detailed}
    async_search_algorithms: Dict[str, Callable] = {"BFS": tot_breadth_first_search, "DFS": tot_depth_first_search}
    
    benchmark_results_dict: Dict[str, ToTResult] = {}
    
    print(f"Starting Async Medical ToT Benchmark with {len(test_methods_list)} methods")
    print("="*60)
    
    for method_combo_str in test_methods_list:
        gen_name, eval_name, search_name = method_combo_str.split("_")
        print(f"\n--- Testing {method_combo_str} (Async) ---")
        
        async_gen_func = async_thought_generators.get(gen_name)
        async_eval_func = async_evaluators.get(eval_name)
        async_search_func = async_search_algorithms.get(search_name)
        
        if not all([async_gen_func, async_eval_func, async_search_func]):
            print(f"Warning: Could not find async methods for {method_combo_str}")
            benchmark_results_dict[method_combo_str] = ToTResult(method_combo_str, [], "Method component not found", 0.0, 0.05, [], 0, {'error': True})
            continue
        
        start_time_val = time.time()
        reasoning_paths_list: List[str] = []
        step_scores_list: List[Any] = [] # Ensure it's initialized
        
        try:
            search_params = { 
                "BFS": {"step_limit": 2, "breadth": 2, "samples_per_step": 2}, # Default BFS params
                "DFS": {"step_limit": 2, "threshold": 5.0, "samples_per_step": 2, "max_paths_to_find": 1} # Default DFS params
            }
            current_search_params = search_params.get(search_name, {})

            reasoning_paths_list, step_scores_list = await async_search_func(
                async_client_param, input_text, medical_knowledge, 
                async_gen_func, async_eval_func, **current_search_params
            )
            
            execution_time_val = time.time() - start_time_val
            final_answer_str = extract_answer_from_paths(reasoning_paths_list, input_text, medical_knowledge) 
            if reasoning_paths_list:
                 final_answer_str = f"Primary Diagnosis Path Analysis: {reasoning_paths_list[0][:800]}..." # More meaningful answer

            confidence_score_val = calculate_confidence_score(reasoning_paths_list, step_scores_list, method_combo_str)
            
            result_obj = ToTResult(
                method_combo_str, reasoning_paths_list, final_answer_str, execution_time_val, 
                confidence_score_val, step_scores_list, 0, 
                {'generator_type': gen_name, 'evaluator_type': eval_name, 'search_type': search_name}
            )
            benchmark_results_dict[method_combo_str] = result_obj
            
            print(f"✓ {method_combo_str} completed in {execution_time_val:.2f}s")
            print(f"  Confidence: {confidence_score_val:.3f}, Overall Quality: {result_obj.get_overall_quality_score():.3f}, Paths: {len(reasoning_paths_list)}")
            
        except Exception as e_inner:
            exec_time_on_error = time.time() - start_time_val
            print(f"✗ Error in {method_combo_str} after {exec_time_on_error:.2f}s: {str(e_inner)}")
            import traceback
            traceback.print_exc() # Add this for more detailed error info during testing
            benchmark_results_dict[method_combo_str] = ToTResult(
                method_combo_str, [], f"Error: {str(e_inner)}", exec_time_on_error, 0.05, [], 0, {'error': True}
            )

    print("\n" + "="*60 + "\nBENCHMARK SUMMARY (Async)\n" + "="*60)
    if benchmark_results_dict:
        sorted_results_list = sorted(
            benchmark_results_dict.items(), 
            key=lambda item: item[1].get_overall_quality_score(), 
            reverse=True
        )
        for i, (method_str, res_obj) in enumerate(sorted_results_list, 1):
            print(f"{i:2d}. {method_str}: Q={res_obj.get_overall_quality_score():.3f} "
                  f"(Conf={res_obj.confidence_score:.3f}, Time={res_obj.execution_time:.1f}s, "
                  f"Paths={len(res_obj.reasoning_paths)})")
    return benchmark_results_dict

async def generate_with_tree_of_thoughts(async_client_param: AsyncOpenAI, input_text_val: str,
                                         response_kg_path: str, response_kg_neighbor: str,
                                         method_str_val: str = "all") -> Union[Dict[str, Any], Tuple[Dict[str, Any], List[str]]]:
    """Async Tree of Thoughts generation. Can run all benchmark methods or a specific one."""
    medical_knowledge_combined = f"Path Evidence: {response_kg_path}\n\nNeighbor Evidence: {response_kg_neighbor}".strip()
    if len(medical_knowledge_combined) < 50: # Arbitrary short length check
        print("Warning: Limited KG medical knowledge provided. ToT results may be suboptimal.")
        medical_knowledge_combined += "\nGeneral medical knowledge should also be applied."
    
    if method_str_val.lower() == "all":
        print("Running comprehensive Async Tree of Thoughts benchmark...")
        benchmark_results_output = await run_tot_benchmark(async_client_param, input_text_val, medical_knowledge_combined)
        
        # Convert ToTResult objects to dictionaries for CSV/downstream use
        output_dict_summary: Dict[str, Any] = {}
        for m_name, tot_res_obj in benchmark_results_output.items():
            output_dict_summary[m_name] = tot_res_obj.get_summary_dict()
            output_dict_summary[m_name]['final_answer'] = tot_res_obj.final_answer # Ensure final answer is included
            output_dict_summary[m_name]['reasoning_paths_summary'] = [p[:100]+"..." for p in tot_res_obj.reasoning_paths[:1]] # Summary of first path
        return output_dict_summary # For "all", return the dict of summaries
    
    else: # Execute a single specified method
        print(f"Running specific async method: {method_str_val}")
        gen_n, eval_n, search_n = method_str_val.split("_") if "_" in method_str_val else method_str_val.split("-")
        if len(method_str_val.split("_")) !=3 and len(method_str_val.split("-")) !=3 : # Basic check
            return {"error": f"Invalid method format: {method_str_val}. Use 'Generator_Evaluator_Search'"}, []

        async_gen_f = {"Sampling": thought_generator_sampling, "Sequential": thought_generator_sequential}.get(gen_n)
        async_eval_f = {"Comprehensive": evaluate_comprehensive, "Comparative": evaluate_comparative_detailed}.get(eval_n)
        async_search_f = {"BFS": tot_breadth_first_search, "DFS": tot_depth_first_search}.get(search_n)

        if not all([async_gen_f, async_eval_f, async_search_f]):
            return {"error": f"Components for method {method_str_val} not found"}, []

        start_t = time.time()
        # Use default parameters for search, or allow them to be passed in
        r_paths, s_scores = await async_search_f(
            async_client_param, input_text_val, medical_knowledge_combined, async_gen_f, async_eval_f 
        )
        exec_t = time.time() - start_t
        
        # For single method, use the final_answer from ToTResult for consistency
        # Create a temporary ToTResult to get a synthesized final_answer
        # Note: This is a bit circular; ideally, extract_answer_from_paths is the primary synthesizer
        temp_final_ans = extract_answer_from_paths(r_paths, input_text_val, medical_knowledge_combined)
        if r_paths: temp_final_ans = f"Primary Diagnosis Path suggests: {r_paths[0][:800]}..."


        conf_score = calculate_confidence_score(r_paths, s_scores, method_str_val)
        
        single_method_result_dict = {
            'final_answer': temp_final_ans, 'reasoning_paths': r_paths,
            'confidence_score': conf_score, 'execution_time': exec_t,
            'method_used': method_str_val, 'step_scores': s_scores
        }
        return single_method_result_dict, r_paths # Return dict and paths for single method


#--------------------------------------
#--------------------------------------


if __name__ == "__main__":
    uri = "bolt://3.238.68.36:7687"
    username = "neo4j"
    password = "term-perforators-accounts"
    YOUR_OPENAI_KEY=os.environ['OPENAI_API_KEY']
    # 在文件开头添加
    from openai import OpenAI
    client = OpenAI(api_key=YOUR_OPENAI_KEY)  # 替换你的API密钥
    async_client = AsyncOpenAI(api_key=YOUR_OPENAI_KEY) 
    os.environ['OPENAI_API_KEY']= YOUR_OPENAI_KEY  # 这行保留
    chat = ChatOpenAI(model="gpt-4o", openai_api_key=YOUR_OPENAI_KEY)

    from neo4j import GraphDatabase

    # 正确配置驱动
    driver = GraphDatabase.driver(
        uri,
        auth=(username, password),
        
    )
    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        with driver.session() as session:
            result = session.run("RETURN 'Hello Neo4j' AS message")
            print(result.single()["message"])  # 应输出 "Hello Neo4j"
        driver.close()
        print("连接成功！")
    except Exception as e:
        print(f"连接失败: {e}")

    driver = GraphDatabase.driver(uri, auth=(username, password))
    session = driver.session()

    ##############################build KG 

    with session.begin_transaction() as tx:
        result = tx.run("MATCH (n) RETURN COUNT(n) AS count")
        node_count = result.single()["count"]
    if node_count == 0:
        print("知识图谱为空，开始重新构建...")
        df = pd.read_csv('./data/chatdoctor5k/train.txt', sep='\t', header=None, names=['head', 'relation', 'tail'])

        for index, row in df.iterrows():
            head_name = row['head']
            tail_name = row['tail']
            relation_name = row['relation']

            query = (
                "MERGE (h:Entity { name: $head_name }) "
                "MERGE (t:Entity { name: $tail_name }) "
                "MERGE (h)-[r:`" + relation_name + "`]->(t)"
            )
            session.run(query, head_name=head_name, tail_name=tail_name, relation_name=relation_name)
        print("知识图谱构建完成！")
    else:
        print(f"已有 {node_count} 个节点，直接使用已有知识图谱，不再重新构建。")


    re1 = r'The extracted entities are (.*?)<END>'
    re2 = r"The extracted entity is (.*?)<END>"
    re3 = r"<CLS>(.*?)<SEP>"
    all_data = []

    with open('./data/chatdoctor5k/entity_embeddings.pkl','rb') as f1:
        entity_embeddings = pickle.load(f1)
    
        
    with open('./data/chatdoctor5k/keyword_embeddings.pkl','rb') as f2:
        keyword_embeddings = pickle.load(f2)

    docs_dir = './data/chatdoctor5k/document'

    docs = []
    for file in os.listdir(docs_dir):
        with open(os.path.join(docs_dir, file), 'r', encoding='utf-8') as f:
            doc = f.read()
            docs.append(doc)
   
    with open("./data/chatdoctor5k/NER_chatgpt.json", "r") as f:
        for line in f.readlines()[40:41]:
            x = json.loads(line)
            input = x["qustion_output"]
            input = input.replace("\n","")
            input = input.replace("<OOS>","<EOS>")
            input = input.replace(":","") + "<END>"
            input_text = re.findall(re3,input)
            
            if input_text == []:
                continue
            print('Question:\n',input_text[0])

            output = x["answer_output"]
            output = output.replace("\n","")
            output = output.replace("<OOS>","<EOS>")
            output = output.replace(":","") + "<END>"
            output_text_reference = re.findall(re3,output)
            # print(output_text[0])

                 
            question_kg = re.findall(re1,input)
            if len(question_kg) == 0:
                question_kg = re.findall(re2,input)
                if len(question_kg) == 0:
                    print("<Warning> no entities found", input)
                    continue
            question_kg = question_kg[0].replace("<END>","").replace("<EOS>","")
            question_kg = question_kg.replace("\n","")
            question_kg = question_kg.split(", ")
            # print("question_kg",question_kg)

            answer_kg = re.findall(re1,output)
            if len(answer_kg) == 0:
                answer_kg = re.findall(re2,output)
                if len(answer_kg) == 0:
                    print("<Warning> no entities found", output)
                    continue
            answer_kg = answer_kg[0].replace("<END>","").replace("<EOS>","")
            answer_kg = answer_kg.replace("\n","")
            answer_kg = answer_kg.split(", ")
            print(answer_kg)

            
            match_kg = []
            entity_embeddings_emb = pd.DataFrame(entity_embeddings["embeddings"])
           

            for kg_entity in question_kg:
                
                keyword_index = keyword_embeddings["keywords"].index(kg_entity)
                kg_entity_emb = np.array(keyword_embeddings["embeddings"][keyword_index])

                cos_similarities = cosine_similarity_manual(entity_embeddings_emb, kg_entity_emb)[0]
                max_index = cos_similarities.argmax()
                          
                match_kg_i = entity_embeddings["entities"][max_index]
                while match_kg_i.replace(" ","_") in match_kg:
                    cos_similarities[max_index] = 0
                    max_index = cos_similarities.argmax()
                    match_kg_i = entity_embeddings["entities"][max_index]

                match_kg.append(match_kg_i.replace(" ","_"))
            # print('match_kg',match_kg)

            # # 4. neo4j knowledge graph path finding
            if len(match_kg) != 1 or 0:
                start_entity = match_kg[0]
                candidate_entity = match_kg[1:]
                
                result_path_list = []
                while 1:
                    flag = 0
                    paths_list = []
                    while candidate_entity != []:
                        end_entity = candidate_entity[0]
                        candidate_entity.remove(end_entity)                        
                        paths,exist_entity = find_shortest_path(start_entity, end_entity,candidate_entity)
                        path_list = []
                        if paths == [''] or paths == []:
                            flag = 1
                            if candidate_entity == []:
                                flag = 0
                                break
                            start_entity = candidate_entity[0]
                            candidate_entity.remove(start_entity)
                            break
                        else:
                            for p in paths:
                                path_list.append(p.split('->'))
                            if path_list != []:
                                paths_list.append(path_list)
                        
                        if exist_entity != {}:
                            try:
                                candidate_entity.remove(exist_entity)
                            except:
                                continue
                        start_entity = end_entity
                    result_path = combine_lists(*paths_list)
                
                
                    if result_path != []:
                        result_path_list.extend(result_path)                
                    if flag == 1:
                        continue
                    else:
                        break
                    
                start_tmp = []
                for path_new in result_path_list:
                
                    if path_new == []:
                        continue
                    if path_new[0] not in start_tmp:
                        start_tmp.append(path_new[0])
                
                if len(start_tmp) == 0:
                        result_path = {}
                        single_path = {}
                else:
                    if len(start_tmp) == 1:
                        result_path = result_path_list[:5]
                    else:
                        result_path = []
                                                  
                        if len(start_tmp) >= 5:
                            for path_new in result_path_list:
                                if path_new == []:
                                    continue
                                if path_new[0] in start_tmp:
                                    result_path.append(path_new)
                                    start_tmp.remove(path_new[0])
                                if len(result_path) == 5:
                                    break
                        else:
                            count = 5 // len(start_tmp)
                            remind = 5 % len(start_tmp)
                            count_tmp = 0
                            for path_new in result_path_list:
                                if len(result_path) < 5:
                                    if path_new == []:
                                        continue
                                    if path_new[0] in start_tmp:
                                        if count_tmp < count:
                                            result_path.append(path_new)
                                            count_tmp += 1
                                        else:
                                            start_tmp.remove(path_new[0])
                                            count_tmp = 0
                                            if path_new[0] in start_tmp:
                                                result_path.append(path_new)
                                                count_tmp += 1

                                        if len(start_tmp) == 1:
                                            count = count + remind
                                else:
                                    break

                    try:
                        single_path = result_path_list[0]
                    except:
                        single_path = result_path_list
                    
            else:
                result_path = {}
                single_path = {}            
            # print('result_path',result_path)
            
            

            # # 5. neo4j knowledge graph neighbor entities
            neighbor_list = []
            neighbor_list_disease = []
            for match_entity in match_kg:
                disease_flag = 0
                neighbors,disease = get_entity_neighbors(match_entity,disease_flag)
                neighbor_list.extend(neighbors)

                while disease != []:
                    new_disease = []
                    for disease_tmp in disease:
                        if disease_tmp in match_kg:
                            new_disease.append(disease_tmp)

                    if len(new_disease) != 0:
                        for disease_entity in new_disease:
                            disease_flag = 1
                            neighbors,disease = get_entity_neighbors(disease_entity,disease_flag)
                            neighbor_list_disease.extend(neighbors)
                    else:
                        for disease_entity in disease:
                            disease_flag = 1
                            neighbors,disease = get_entity_neighbors(disease_entity,disease_flag)
                            neighbor_list_disease.extend(neighbors)
            if len(neighbor_list)<=5:
                neighbor_list.extend(neighbor_list_disease)

            # print("neighbor_list",neighbor_list)


            # 6. knowledge gragh path based prompt generation
            if len(match_kg) != 1 or 0:
                response_of_KG_list_path = []
                if result_path == {}:
                    response_of_KG_list_path = []
                else:
                    result_new_path = []
                    for total_path_i in result_path:
                        path_input = "->".join(total_path_i)
                        result_new_path.append(path_input)
                    
                    path = "\n".join(result_new_path)
                    response_of_KG_list_path = prompt_path_finding(path)
                    if is_unable_to_answer(response_of_KG_list_path):
                        response_of_KG_list_path = prompt_path_finding(path)
                    # print("response_of_KG_list_path",response_of_KG_list_path)
            else:
                response_of_KG_list_path = '{}'

            response_single_path = prompt_path_finding(single_path)
            if is_unable_to_answer(response_single_path):
                response_single_path = prompt_path_finding(single_path)

            # # 7. knowledge gragh neighbor entities based prompt generation   
            response_of_KG_list_neighbor = []
            neighbor_new_list = []
            for neighbor_i in neighbor_list:
                neighbor = "->".join(neighbor_i)
                neighbor_new_list.append(neighbor)

            if len(neighbor_new_list) > 5:

                neighbor_input = "\n".join(neighbor_new_list[:5])
            response_of_KG_neighbor = prompt_neighbor(neighbor_input)
            if is_unable_to_answer(response_of_KG_neighbor):
                response_of_KG_neighbor = prompt_neighbor(neighbor_input)
            # print("response_of_KG_neighbor",response_of_KG_neighbor)

#------------------------------------------------------------#
            # 8. prompt-based medical diaglogue answer generation
            output_all_mindmap = final_answer(input_text[0],response_of_KG_list_path,response_of_KG_neighbor)
            if is_unable_to_answer(output_all_mindmap):
                output_all_mindmap = final_answer(input_text[0],response_of_KG_list_path,response_of_KG_neighbor)
            
            print('\nMindMap:\n',output_all_mindmap)

            ### mindmap-nokg####
            output_all_mindmap_nokg = final_answer_nokg(input_text[0])
            if is_unable_to_answer(output_all_mindmap_nokg):
                output_all_mindmap_nokg = final_answer_nokg(input_text[0])
            
            print('\nMindMap:\n',output_all_mindmap_nokg)

            ### 10. Experiment 2: document retrieval + bm25
            document_dir = "./data/chatdoctor5k/document"
            document_paths = [os.path.join(document_dir, f) for f in os.listdir(document_dir)]

            corpus = []
            for path in document_paths:
                with open(path, "r", encoding="utf-8") as f:
                    corpus.append(f.read().lower().split())

            dictionary = corpora.Dictionary(corpus)
            bm25_model = BM25Okapi(corpus)

            bm25_corpus = [bm25_model.get_scores(doc) for doc in corpus]
            bm25_index = SparseMatrixSimilarity(bm25_corpus, num_features=len(dictionary))

            query = input_text[0]
            query_tokens = query.lower().split()
            tfidf_model = TfidfModel(dictionary=dictionary, smartirs='bnn')
            tfidf_query = tfidf_model[dictionary.doc2bow(query_tokens)]
            best_document_index, best_similarity = 0, 0  

            bm25_scores = bm25_index[tfidf_query]
            for i, score in enumerate(bm25_scores):
                if score > best_similarity:
                    best_similarity = score
                    best_document_index = i

            with open(document_paths[best_document_index], "r", encoding="utf-8") as f:
                best_document_content = f.read()

            document_bm25_result = prompt_document(input_text[0],best_document_content)
            if is_unable_to_answer(document_bm25_result):
                document_bm25_result = prompt_document(input_text[0],best_document_content)
            
            print('\nBM25_retrieval:\n',document_bm25_result)

            ### 11. Experiment 3: document + embedding retrieval
            model = Word2Vec.load("./data/chatdoctor5k/word2vec.model")
            ques_vec = np.mean([model.wv[token] for token in input_text[0].split()], axis=0)
            similarities = []
            for doc in docs:
                doc_vec = np.mean([model.wv[token] for token in doc.split()], axis=0)
                similarity = cosine_similarity([ques_vec], [doc_vec])[0][0]
                similarities.append(similarity)

            max_index = np.argmax(similarities)
            most_similar_doc = docs[max_index]

            ### 12. Experiment 4: kg retrieval
            kg_retrieval = prompt_document(input_text[0],response_single_path)
            if is_unable_to_answer(kg_retrieval):
                kg_retrieval = prompt_document(input_text[0],response_single_path)
            print('\nKG_retrieval:\n',kg_retrieval)

            kg_retrieval_multipath = prompt_document(input_text[0],response_of_KG_list_path)
            if is_unable_to_answer(kg_retrieval):
                kg_retrieval_multipath = prompt_document(input_text[0],response_of_KG_list_path)
            print('\nkg_retrieval_multipath:\n',kg_retrieval_multipath)

            # # # 9. Experiment 5: AutoGen
            base_patient_question = input_text[0]
            kg_path_context = response_of_KG_list_path if response_of_KG_list_path else "No path-based evidence provided."
            kg_neighbor_context = response_of_KG_neighbor if response_of_KG_neighbor else "No neighbor-based evidence provided."

            # # Create experiment instance
            experiment = MedicalQAAutoGenExperiment(YOUR_OPENAI_KEY)
            
            # Prepare standardized input for agents
            common_user_input = experiment.create_medical_prompt_template(
                base_patient_question, 
                kg_path_context, 
                kg_neighbor_context
            )
            
            print('=== AutoGen Single-Agent Experiment ===')
            autogen_single_result = "Error: No result generated"
            try:
                autogen_single_result = experiment.run_single_agent_experiment(common_user_input)
                print(f'Single-Agent Result:\n{autogen_single_result}\n')
            except Exception as e:
                autogen_single_result = f"Single-agent error: {str(e)}"
                logger.error(f"Single-agent experiment failed: {e}", exc_info=True)
                print(f'Single-Agent Result: {autogen_single_result}\n')

            try:
                autogen_single_result = experiment.run_single_agent_experiment(common_user_input)
                print(f'Single-Agent Result:\n{autogen_single_result}\n')
            except Exception as e:
                autogen_single_result = f"Single-agent error: {str(e)}"
                logger.error(f"Single-agent experiment failed: {e}", exc_info=True)
                print(f'Single-Agent Result: {autogen_single_result}\n')


            print('\n=== AutoGen Multi-Agent Experiment ===')
            autogen_multi_result = "Error: No result generated"
            try:
                autogen_multi_result = experiment.run_multi_agent_experiment(common_user_input)
                print(f'Multi-Agent Result:\n{autogen_multi_result}\n')
            except Exception as e:
                autogen_multi_result = f"Multi-agent error: {str(e)}"
                logger.error(f"Multi-agent experiment failed: {e}", exc_info=True)
                print(f'Multi-Agent Result: {autogen_multi_result}\n')

            ##------no kg---------###

            common_user_input_nokg = experiment.create_medical_prompt_template(
                base_patient_question,
                path_evidence="",
                neighbor_evidence="")
            
            autogen_single_result_nokg = "Error: No result generated"
            try:
                autogen_single_result_nokg = experiment.run_single_agent_experiment(common_user_input_nokg)
                print(f'Single-Agent Result:\n{autogen_single_result_nokg}\n')
            except Exception as e:
                autogen_single_result_nokg = f"Single-agent error: {str(e)}"
                logger.error(f"Single-agent experiment failed: {e}", exc_info=True)
                print(f'Single-Agent Result: {autogen_single_result_nokg}\n')

            autogen_multi_result_nokg = "Error: No result generated"
            try:
                autogen_multi_result_nokg = experiment.run_multi_agent_experiment(common_user_input_nokg)
                print(f'Multi-Agent Result:\n{autogen_multi_result_nokg}\n')
            except Exception as e:
                autogen_multi_result_nokg = f"Multi-agent error: {str(e)}"
                logger.error(f"Multi-agent experiment failed: {e}", exc_info=True)
                print(f'Multi-Agent Result: {autogen_multi_result_nokg}\n')
            ### 14. 新增实验方法：KG+Self-Consistency
            try:
                print("\nGenerating KG+Self-Consistency response...")
                sc_consensus, sc_candidates = generate_with_self_consistency(
                    input_text[0], 
                    response_of_KG_list_path,
                    response_of_KG_neighbor,
                    num_samples=5  # Use 3 samples for better consistency vs. speed tradeoff
                )
                print('\nKG+Self-Consistency result:\n', sc_consensus[:])
            except Exception as e:
                print(f"Error in Self-Consistency generation: {e}")
                sc_consensus = "Error generating self-consistency response"

            ### 14. Experiment 6: Zero-shot Prompting
            print('\n=== Zero-shot Prompting ===')
            zero_shot_prompt = zero_shot_prompt_template(str(input_text[0]))

            # GPT-4o-mini Zero-shot
            try:
                zero_shot_4o_mini = chat_4o_mini(zero_shot_prompt)
            except:
                sleep(40)
                zero_shot_4o_mini = chat_4o_mini(zero_shot_prompt)
            print('\nZero-shot GPT-4o-mini:', zero_shot_4o_mini)

            # GPT-4 Zero-shot
            try:
                zero_shot_gpt4 = chat_4(zero_shot_prompt)
            except:
                zero_shot_gpt4 = chat_4(zero_shot_prompt)
            print('\nZero-shot GPT-4:', zero_shot_gpt4)

            
            #######zeorshot-kg###########
            zero_shot_prompt_with_kg = zero_shot_prompt_template_with_kg(str(input_text[0]),response_of_KG_list_path,response_of_KG_neighbor)
            try:
                zero_shot_4o_mini_kg = chat_4o_mini(zero_shot_prompt_with_kg)
            except:
                sleep(40)
                zero_shot_4o_mini_kg = chat_4o_mini(zero_shot_prompt_with_kg)
            print('\nZero-shot GPT-4o-mini:', zero_shot_4o_mini_kg)

            # GPT-4 Zero-shot
            try:
                zero_shot_gpt4_kg = chat_4(zero_shot_prompt_with_kg)
            except:
                zero_shot_gpt4_kg = chat_4(zero_shot_prompt_with_kg)
            print('\nZero-shot GPT-4:', zero_shot_gpt4_kg)



















            ### 15. Experiment 7: One-shot Prompting
            print('\n=== One-shot Prompting ===')
            one_shot_prompt = one_shot_prompt_template(str(input_text[0]))

            # GPT-4o-mini One-shot
            try:
                one_shot_4o_mini = chat_4o_mini(one_shot_prompt)
            except:
                sleep(40)
                one_shot_4o_mini = chat_4o_mini(one_shot_prompt)
            print('\nOne-shot GPT-4o-mini:', one_shot_4o_mini)

            # GPT-4 One-shot
            try:
                one_shot_gpt4 = chat_4(one_shot_prompt)
            except:
                one_shot_gpt4 = chat_4(one_shot_prompt)
            print('\nOne-shot GPT-4:', one_shot_gpt4)

            ### 16. Experiment 8: Few-shot Prompting
            print('\n=== Few-shot Prompting ===')
            few_shot_prompt = few_shot_prompt_template(str(input_text[0]))

            # GPT-4o-mini Few-shot
            try:
                few_shot_4o_mini = chat_4o_mini(few_shot_prompt)
            except:
                sleep(40)
                few_shot_4o_mini = chat_4o_mini(few_shot_prompt)
            print('\nFew-shot GPT-4o-mini:', few_shot_4o_mini)

            # GPT-4 Few-shot
            try:
                few_shot_gpt4 = chat_4(few_shot_prompt)
            except:
                few_shot_gpt4 = chat_4(few_shot_prompt)
            print('\nFew-shot GPT-4:', few_shot_gpt4)

            ### 17. Experiment 9: Chain-of-thought Prompting
            print('\n=== Chain-of-thought Prompting ===')
            cot_prompt = chain_of_thought_prompt_template(str(input_text[0]))

            # GPT-4o-mini Chain-of-thought
            try:
                cot_4o_mini = chat_4o_mini(cot_prompt)
            except:
                sleep(40)
                cot_4o_mini = chat_4o_mini(cot_prompt)
            print('\nCoT GPT-4o-mini:', cot_4o_mini)

            # GPT-4 Chain-of-thought
            try:
                cot_gpt4 = chat_4(cot_prompt)
            except:
                cot_gpt4 = chat_4(cot_prompt)
            print('\nCoT GPT-4:', cot_gpt4)
#----------------------------#
            # 15. 新增实验方法：KG+Tree-of-Thoughts
            try:
                print("Starting KG+Tree-of-Thoughts benchmark (Async)...")
                # Call the asynchronous function using asyncio.run() and pass the async_client
                tot_all_benchmark_results = asyncio.run(generate_with_tree_of_thoughts(
                    async_client,
                    input_text[0] if isinstance(input_text, list) else input_text,
                    response_of_KG_list_path,
                    response_of_KG_neighbor,
                    "all"  # This is correct as it's the 5th positional for method_str_val
                ))
                
                print('\nKG+Tree-of-Thoughts (All Benchmark Results):\n')
                
                # This printing logic should work as combo_data is the summary dictionary
                if isinstance(tot_all_benchmark_results, dict) and tot_all_benchmark_results:
                    for combo_name, results_data in tot_all_benchmark_results.items():
                        if isinstance(results_data, dict) and 'final_answer' in results_data:
                            print(f"  ✓ Method: {combo_name}")
                            print(f"    Confidence: {results_data.get('confidence_score', 0.0):.2f}")
                            print(f"    Time: {results_data.get('execution_time', 0.0):.2f}s")
                            print(f"    Num Paths: {results_data.get('num_paths', 0)}") # Added for clarity
                        else:
                            print(f"  ✗ Method: {combo_name} - Error or invalid data format in results_data")
                            # print(f"    Problematic results_data: {results_data}") # For debugging
                else:
                    print("  No valid results returned or tot_all_benchmark_results is not a populated dictionary.")

            except Exception as e:
                print(f"Error in KG+Tree-of-Thoughts: {str(e)}")
                import traceback
                traceback.print_exc() # Print full traceback for better debugging
                tot_all_benchmark_results = {} # Ensure it's a dict for consistent handling below

            current_data_row = {
                'Question': input_text[0],
                'Label': output_text_reference[0],
                'Mindmap': (output_all_mindmap),
                'MindmapwithoutKG':(output_all_mindmap_nokg),
                'BM25_retrieval':(document_bm25_result),
                'Embedding_retrieval': (document_embedding_result),
                'KG_retrieval': (kg_retrieval),
                'KG_retrieval_multipath':(kg_retrieval_multipath),
                'KG_self-consistency': (sc_consensus),
                'Zero_shot_GPT4o_mini_kg': (zero_shot_4o_mini_kg),
                'zero_shot_gpt4_kg':(zero_shot_gpt4_kg),
                'Zero_shot_GPT4o_mini': (zero_shot_4o_mini),
                'Zero_shot_GPT4o': (zero_shot_gpt4),
                'One_shot_GPT4o_mini': (one_shot_4o_mini),
                'One_shot_GPT4o': (one_shot_gpt4),
                'Few_shot_GPT4o_mini': (few_shot_4o_mini),
                'Few_shot_GPT4o': (few_shot_gpt4),
                'CoT_GPT4o_mini': (cot_4o_mini),
                'CoT_GPT4o': (cot_gpt4),
                'AutoGen_Single_Result':(autogen_single_result),
                'AutoGen_Single_Result_nokg': (autogen_single_result_nokg),
                'AutoGen_Multi_Result':(autogen_multi_result),
                'Multi-Agent Result_nokg': (autogen_multi_result_nokg),
                # 'TaskWeaver_Simple_Result':(taskweaver_simple_result),
                # 'TaskWeaver_Multi_Plugin_Result': (taskweaver_multi_plugin_result)
            }
###@@@
            # 将ToT结果添加到current_data_row
            if isinstance(tot_all_benchmark_results, dict) and tot_all_benchmark_results:
                for combo_name, combo_data in tot_all_benchmark_results.items():
                    # combo_data is the summary dictionary from ToTResult.get_summary_dict()
                    safe_combo_name = combo_name.replace(" ", "_").replace("-", "_")
                    base_column = f'KG_ToT_{safe_combo_name}'
                    
                    if isinstance(combo_data, dict) and 'final_answer' in combo_data:
                        # 添加主要结果
                        current_data_row[base_column] = combo_data['final_answer']
                        current_data_row[f'{base_column}_confidence'] = combo_data.get('confidence_score', 0.0)
                        current_data_row[f'{base_column}_time'] = combo_data.get('execution_time', 0.0)
                        # Correctly get the number of paths from the summary dictionary
                        current_data_row[f'{base_column}_paths'] = combo_data.get('num_paths', 0) 
                    else:
                        current_data_row[base_column] = "Error: Invalid ToT combo_data format"
                        # print(f"Debug: Invalid combo_data for {combo_name}: {combo_data}") # For debugging
            else:
                current_data_row['KG_ToT_Status'] = "Failed to retrieve benchmark results or empty results"
            
            all_data.append(current_data_row)
            # ***** END OF MODIFICATION *****

    # 将收集的所有数据转换为DataFrame并保存为CSV
    df = pd.DataFrame(all_data)
    df.to_csv('output.csv', index=False, encoding='utf-8', quoting=csv.QUOTE_ALL)
    print("CSV file has been saved successfully!")