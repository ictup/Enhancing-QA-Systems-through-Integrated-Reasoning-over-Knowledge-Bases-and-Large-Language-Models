#!/usr/bin/env python
import os
from dotenv import load_dotenv
from src.agents.pipeline import run_pipeline

if __name__ == "__main__":
    load_dotenv()
    run_pipeline()
