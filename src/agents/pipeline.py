import os
from dotenv import load_dotenv

def run_pipeline():
    load_dotenv()
    # TODO: wire up retrieval -> KG reasoning -> prompts -> LLM generation -> evaluation
    print("Pipeline entrypoint placeholder. Configure in scripts/run_agent.py")

if __name__ == '__main__':
    run_pipeline()
