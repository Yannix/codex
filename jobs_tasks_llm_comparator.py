# -*- coding: utf-8 -*-
"""
Created on Sat May 31 17:23:14 2025

@author: YanniX
"""

from openai import OpenAI
from mistralai import Mistral
import time
import os
import json
import re
import ast
import pandas as pd
from dotenv import load_dotenv
# os.path.dirname(os.path.abspath(__file__))

#  os.getcwd()

script_start_time = time.time()

load_dotenv()


## Read API access keys

OPEN_AI_KEY = os.getenv('OPEN_AI_KEY')
GEMINI_KEY = os.getenv('GEMINI_KEY')
DEEPSEEK_KEY = os.getenv('DEEPSEEK_KEY')
MISTRAL_AI_KEY = os.getenv('MISTRAL_AI_KEY')
QWEN_AI_KEY = os.getenv('QWEN_AI_KEY')


## Define LLMs APIs properties

openai_llms_list = [ {'identifier' : 'gpt-4.1',         'name' : 'GPT 4.1',         'base_url' : None, 'provider' : 'Openai', 'api_key' : OPEN_AI_KEY },
                     {'identifier' : 'o4-mini',         'name' : 'o4 Mini',         'base_url' : None, 'provider' : 'Openai', 'api_key' : OPEN_AI_KEY },
                     #{'identifier' : 'o3',              'name' : 'o3',              'base_url' : None, 'provider' : 'Openai', 'api_key' : OPEN_AI_KEY },
                     #{'identifier' : 'gpt-4.5-preview', 'name' : 'GPT 4.5 Preview', 'base_url' : None, 'provider' : 'Openai', 'api_key' : OPEN_AI_KEY },
                     {'identifier' : 'gpt-4o',          'name' : 'GPT 4o',          'base_url' : None, 'provider' : 'Openai', 'api_key' : OPEN_AI_KEY },
                     
                     {'identifier' : 'deepseek-chat',       'name' : 'DeepSeek-V3-0324',    'base_url' : 'https://api.deepseek.com', 'provider' : 'DeepSeek',   'api_key' : DEEPSEEK_KEY },
                     {'identifier' : 'deepseek-reasoner',   'name' : 'DeepSeek-R1-0528',    'base_url' : 'https://api.deepseek.com', 'provider' : 'DeepSeek',   'api_key' : DEEPSEEK_KEY },
                     
                     {'identifier' : 'gemini-2.5-pro-preview-05-06',    'name' : 'Gemini 2.5 Pro Preview',          'base_url' : 'https://generativelanguage.googleapis.com/v1beta/openai/', 'provider' : 'Google', 'api_key' : GEMINI_KEY },
                     {'identifier' : 'gemini-2.5-flash-preview-05-20',  'name' : 'Gemini 2.5 Flash Preview 05-20',  'base_url' : 'https://generativelanguage.googleapis.com/v1beta/openai/', 'provider' : 'Google', 'api_key' : GEMINI_KEY },
                     {'identifier' : 'gemini-2.0-flash',                'name' : 'Gemini 2.0 Flash',                'base_url' : 'https://generativelanguage.googleapis.com/v1beta/openai/', 'provider' : 'Google', 'api_key' : GEMINI_KEY },
                     {'identifier' : 'gemini-2.0-flash-lite',           'name' : 'Gemini 2.0 Flash-Lite',           'base_url' : 'https://generativelanguage.googleapis.com/v1beta/openai/', 'provider' : 'Google', 'api_key' : GEMINI_KEY },
                     
                     {'identifier' : 'qwen-max-latest',    'name' : 'Qwen-Max',     'base_url' : 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1', 'provider' : 'Alibaba/Qwen', 'api_key' : QWEN_AI_KEY },
                     {'identifier' : 'qwen-plus-latest',   'name' : 'Qwen-Plus',    'base_url' : 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1', 'provider' : 'Alibaba/Qwen', 'api_key' : QWEN_AI_KEY },
                     {'identifier' : 'qwen-turbo-latest',  'name' : 'Qwen-Turbo',   'base_url' : 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1', 'provider' : 'Alibaba/Qwen', 'api_key' : QWEN_AI_KEY },
                    
                     ]


mistral_llms_list = [   {'identifier' : 'mistral-medium-2505',  'name' : 'Mistral Medium',  'provider' : 'Mistral', 'api_key' : MISTRAL_AI_KEY },
                        {'identifier' : 'mistral-large-2411',   'name' : 'Mistral Large',   'provider' : 'Mistral', 'api_key' : MISTRAL_AI_KEY },
                        {'identifier' : 'mistral-small-2503',   'name' : 'Mistral Small',   'provider' : 'Mistral', 'api_key' : MISTRAL_AI_KEY },
                    ]



## Build the prompt following Augustin's idea

TOOLS_TECHNOLOGIES = """
* Large Language Models - LLMs :
    ** Anthropic Claude
    ** OpenAI ChatGPT (GPT 4.5, GTP-4o)
    ** Google Gemini 3.5
    ** Grok 3
    ** Cohere Command R
    ** DeepSeek R1
    ** Llama 3 (1B et 3B)
    ** Llama (11B et 90B)
    ** Mistral AI (Mistral Small 3)
    
* AI Agents Orchestrator and Low-Code/No-Code - Tools :
    ** AWS App Studio
    ** Flowise AI
    ** N8N
    ** AnythingLLM
    ** Cursor AI
    ** Langchain
    ** LLamaindex
    
* Infrastructure - Operating systems :
    ** Windows Server
    ** Linux (OS)
    ** Ubuntu Linux
    ** Red Hat Enterprise Linux (RHEL)
    ** Amazon Linux AMI

* Infrastructure - Runtime environment :
    ** Kubernetes
    
* Infrastructure - Container Orchestration :
    ** Docker
    
* Infrastructure - Service Mesh :
    ** Istio
    
* Infrastructure - Content Delivery Network :
    ** AWS CloudFront
    
* Infrastructure - Secrets Management :
    ** Vault
    
* Infrastructure - Logs :
    ** Splunk

* Infrastructure - Traces & Metrics :
    ** Datadog

* CI / CD - Code Repository :
    ** BitBucket
    ** GitHub

* CI / CD - Artifact Repository :
    ** Nexus
    ** Artifactory

* CI / CD - Continuous Integration :
    ** Jenkins
    ** GitHub Actions

* CI / CD - CD Config :
    ** Helm
    
* CI / CD - CD Engine :
    ** Harness
    ** GitHub Actions + ArgoCD

* CI / CD - Static Code Analysis :
    ** SonarQube
    
* CI / CD - Artifact Security Analysis :
    ** Snyk
    
* CI / CD - Container Security Analysis :
    ** Aquasec
    
* API - API Documentation Standard :
    ** OpenAPI 3.0.0

* Backend - Runtime infrastructure :
    ** AWS EKS
    ** Kubernetes

* Backend - Security :
    ** Oauth 2.0 Okta
    ** Cloudflare (SSL & Proxy)

* Backend - Architecture :
    ** Micro-service
    ** Event-Driven
    ** Domain-Driven Design

* Backend - Runtime environment :
    ** Java
    ** Python (Flask, Django, FastAPI)
    ** PHP (Symfony, Laravel, Codeigniter, Slim)
    ** NodeJS (Express, Next, Angular)

* Backend - Architecture pattern :
    ** Model-View-Controller (MVC)
    ** Model-View-ViewModel (MVVM)

* Backend - Static Code Analysis :
    ** REST Client
    ** Feign Client

* Backend - Builder / Dependency Manager :
    ** Maven
    
* Frontend - Implementation languages :
    ** Typescript
    ** Javascript
    ** Vue.js
    ** Flutter
    ** React Native

* Frontend - CSS Frameworks :
    ** Tailwind css
    ** Bootstrap
    ** Material UI
        
* Frontend - Security :
    ** Oauth 2.0 Okta (external authentication)
    ** Cloudflare (SSL & Proxy)
    ** Active Directory (internal authentication)

* Web library :
    ** React
    
* Builder / Packager :
    ** Webpack

* Dependency Manager :
    ** NPM
    
* Storage - Databases : 
    ** Oracle
    ** MS SQL Server
    ** MySQL
    ** Firestore
    ** PostgreSQL
    ** MongoDB
    ** MariaDB
    ** Cassandra

* Storage - Vectorial databases : 
    ** Pinecone
    ** ChromaDB
    ** PGVector
    ** AWS OpenSearch
    ** OpenSearch
    ** Faiss
    ** Valkey
    ** Milvus
    ** Qdrant

* Storage - Cost-effective File Storage :
    ** AWS S3
    ** Google drive
    ** Azure Blob Storage
    
* Storage - Streaming (Event-Driven) :
    ** Kafka
    ** AWS SQS

* Storage - Queuing :
    ** IBM MQ
    ** Rabbit MQ

* Storage - Caching :
    ** Redis
    ** AWS DynamoDB
    ** Memcache
    
* Cloud Provider :
    ** Digital Ocean
    ** Gcloud
    ** Hostinger VPS
    ** Contabo VPS
    ** Render
"""

USE_CASES = """
Develop a Web application that is mobile-first and compatible laptop.
This application will run on a robust Linux server, hosted on a cloud and cost-effective system.
Data are persisted on a robust database with replication across two sites.
Application runs behind a firewall with redirection to a load balancer.
Application has three layers: front-end, middleware with API Gateway that expose REST APIs, and back-end.
"""

SYSTEM_PROMPT = """
Tu es un conseiller virtuel expert en identification des meilleures technologies bon-marché et robustes pour faire fonctionner une solution informatique business. 
Ton role est d'aider les clients à trouver les meilleures combinaisons de technologie pour batir un stack technologique. Tu devras lire un use case et tu  fouilleras dans l'ensemble de ta connaissance (et sur internet si c'est possible) en te limitant aux choix technologiques donnes dans la section ### STACK TECHNOS ### pour trouver les stack technologiques qui respondent a cet use case.

Tu détermineras ce stack technologique parmi les technologies classées comme LLMs, orchestrateurs d'agents AI, outils Low-Code/No-Code, infrastructure, outils de CI/CD et de DevSecOps, backend et frontend, bases des données, bases vectorielles et le Cloud et file sharing

* Tu valideras les choix du stack technologique à proposer avec les meilleurs stacks technologiques qui existent dans l'ensemble de ta connaissance (et sur internet si c'est possible).
* Pour formuler un stack technologique pertinent, tu poses des questions ciblées sur les aspects infrastructure, securite des donnees, cloud, intelligence artificielle, communication entre serveur applicatif et base des donnees, reseautique
* Tu expliques les raisons du choix du stack technologique de facon claire et simple, sans jargon, en mettant l'accent sur la valeur réelle que chaque technologie apporte dans le stack.
* Tu es interdit de faire une combinaison des technologies non identifiées ou qui ne cadre pas avec le use case. Tu es aussi interdit de communiquer sur des technologies hors contexte.
* Ton ton est professionnel, empathique, simple et rassurant. Si une situation nécessite l'avis d'un conseiller humain, tu le precises de manière courtoise.
* La liste des technologies acceptables est dans la section ### STACK TECHNOS ###. Le use case est décrit dans la section ### USE CASE ###.
* Ne proposer un stack que quand il est applicable dans le contexte du use case. Si un stack technologique n'est pas applicable répondre par "N/A"
* Ta réponse doit être au format json en plain text et valide, sans caractères additionnels autour, pas de format markdwon. La réponse json doit respecter l'exemple du schema suivant : {  "summary" : "un résumé du use case", 
                                                                                                                                                    "stacks" : [ { "label" : "Large Language Models - LLMs",
                                                                                                                                                                   "choix" : "Gemini 3.5 Pro",
                                                                                                                                                                   "raisons" : "Voici la raison du choix de la technologie Gemini 3.5 Pro..."
                                                                                                                                                                 },
                                                                                                                                                                { "Label" : "AI Agents Orchestrator and Low-Code/No-Code - Tools",
                                                                                                                                                                  "choix' : "Langchain",
                                                                                                                                                                  "raisons' : "Langchain a été choisi parcque ..."
                                                                                                                                                                },
                                                                                                                                                                { "label' : 'Infrastructure - Operating systems",
                                                                                                                                                                  "choix' : 'Red Hat Enterprise Linux",
                                                                                                                                                                  "raisons' : 'Red Hat Enterprise Linux a été recommandé en raison de ..."
                                                                                                                                                                },
                                                                                                                                                                { "label" : "Infrastructure - Runtime environment ",
                                                                                                                                                                  "choix" : "N/A",
                                                                                                                                                                  "raisons" : "Pour ce use case Runtime environment n est pas..."
                                                                                                                                                                },
                                                                                                                                                                ]
                                                                                                                                                    }
"""

SYSTEM_PROMPT = SYSTEM_PROMPT + f"""

### STACK TECHNOS ###
{TOOLS_TECHNOLOGIES}
### STACK TECHNOS ###

"""

USER_PROMPT = f"""

### USE CASE ###
{USE_CASES}
### USE CASE ###

"""

## LLM API queries

llm_responses = []

for i in openai_llms_list :
    
    start_time = time.time()
    
    print(f"LLM Running now : {i['name']}")
    
    if i['api_key'] is None:
        client = OpenAI(api_key=i['api_key'])
    else:
        client = OpenAI(api_key=i['api_key'], base_url=i['base_url'])

    response = client.chat.completions.create(
        model = i['identifier'],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
        stream=False
    )

    response = response.choices[0].message.content
    llm_responses.append((i['name'], response))
    
    end_time = time.time()

    print("LLM time to execute : %s seconds" % (end_time - start_time))
    print("--")
    

for i in mistral_llms_list :
    
    start_time = time.time()
    
    print(f"LLM Running now : {i['name']}")
    
    client = Mistral(api_key=i['api_key'])

    response = client.chat.complete(
        model= i['identifier'],
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ]
    )

    response = response.choices[0].message.content
    llm_responses.append((i['name'], response))
    
    end_time = time.time()

    print("LLM time to execute : %s seconds" % (end_time - start_time))
    print("--")


## Result in nice format

result_table = []

for llm_response in llm_responses:
    llm_stack = llm_response[1].strip()
    llm_stack = llm_stack.strip('```json')
    llm_stack = llm_stack.strip('```')
    llm_stack = llm_stack.strip("\n")
    #llm_stack = re.sub(r"(\s*\{\s*|\s*,\s*)'([^']+)':", r'\1"\2":', llm_stack)
    #llm_stack = ast.literal_eval(json.dumps(llm_stack))
    llm_stack = ast.literal_eval(llm_stack)
    #llm_stack = llm_stack.replace("\'", "\"")
    #llm_stack = json.loads(llm_stack)
    
    
    llm = llm_response[0]
    summary = llm_stack['summary']
    stacks = llm_stack['stacks']
    
    for stack in stacks :
        result_table.append([llm.strip(), summary.strip(), stack['label'].strip(), stack['choix'].strip(), stack['raisons'].strip()])
    

df = pd.DataFrame(data=result_table, columns=['LLM', 'Use case summary', 'Stack', 'Choice', 'Reason'])

df_choice = pd.crosstab(index = df['Stack'], columns = df['LLM'], values = df['Choice'], aggfunc='first')
df_choice = df_choice.fillna('N/A')

df.to_excel(r'C:\Workspace\mayeleai\llm_result_' + pd.to_datetime('now').strftime('%Y_%m_%d_%H_%M') + '.xlsx')
df_choice.to_excel(r'C:\Workspace\mayeleai\llm_result_choice_' + pd.to_datetime('now').strftime('%Y_%m_%d_%H_%M') + '.xlsx')

## LLM as judge - use GPT-4o to determine the best response
judge_system_prompt = "Vous êtes un juge expert chargé de comparer différentes réponses à un même problème et d'identifier celle qui est la plus pertinente, claire et complète.".strip()

judge_messages = [{"role": "system", "content": judge_system_prompt}]

for name, response in llm_responses:
    judge_messages.append({"role": "assistant", "content": f"Réponse de {name} :\n{response}"})

judge_messages.append({"role": "user", "content": "Quelle est la meilleure réponse ? Veuillez répondre au format JSON : {'best_llm': 'nom', 'reason': 'explication courte'}"})

judge_client = OpenAI(api_key=OPEN_AI_KEY)
judge_response = judge_client.chat.completions.create(
    model="gpt-4o",
    messages=judge_messages,
    stream=False
)

best_llm_raw = judge_response.choices[0].message.content.strip()
try:
    best_llm_raw = best_llm_raw.strip('```json').strip('```')
    best_llm = ast.literal_eval(best_llm_raw)
    print(f"LLM gagnant : {best_llm.get('best_llm')}\nRaison : {best_llm.get('reason')}")
except Exception:
    print(best_llm_raw)

script_end_time = time.time()
print("Time to execute the script : %s seconds" % (script_end_time - script_start_time))
