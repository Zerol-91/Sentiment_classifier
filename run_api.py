
import uvicorn
import os
from dotenv import load_dotenv
import sys

load_dotenv()
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == "__main__":

    uvicorn.run(
        "src.api:app",          
        host="0.0.0.0",         
        port=8000,              
        reload=False, 
        log_level="info"        
    )