# Crypto Analysis App --> By Pejman Ebrahimi
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from pydantic import BaseModel
from typing import Dict, List
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Author information
AUTHOR_INFO = {
    "name": "Pejman Ebrahimi",
    "linkedin": "https://www.linkedin.com/in/pejman-ebrahimi-4a60151a7/",
    "github": "https://github.com/arad1367",
    "scholar": "https://scholar.google.com/citations?user=zO-nHd0AAAAJ&hl=en",
    "website": "https://arad1367.github.io/pejman-ebrahimi/",
    "huggingface": "https://huggingface.co/arad1367"
}

# Custom HTML description
description = """
# Cryptocurrency Analysis API

## Created by [Pejman Ebrahimi]({linkedin})

A professional cryptocurrency analysis tool that provides real-time technical analysis using advanced AI models.

### 🔗 Author Links:
* 👨‍💼 [LinkedIn]({linkedin})
* 💻 [GitHub]({github})
* 📚 [Google Scholar]({scholar})
* 🌐 [Personal Website]({website})
* 🤗 [Hugging Face]({huggingface})

### Features:
* Real-time cryptocurrency analysis
* Technical indicators calculation
* AI-powered market insights
* Support for major cryptocurrencies
""".format(**AUTHOR_INFO)

app = FastAPI(
    title="Crypto Analysis API - By Pejman Ebrahimi",
    description=description,
    version="1.0.0",
    contact={
        "name": "Pejman Ebrahimi",
        "url": AUTHOR_INFO["linkedin"],
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    }
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define supported cryptocurrencies with metadata
SUPPORTED_CRYPTOCURRENCIES = {
    "BTC": {
        "name": "Bitcoin",
        "description": "The first and largest cryptocurrency by market cap",
        "decimal_places": 2
    },
    "ETH": {
        "name": "Ethereum",
        "description": "Leading smart contract platform",
        "decimal_places": 2
    },
    "BNB": {
        "name": "Binance Coin",
        "description": "Native token of Binance ecosystem",
        "decimal_places": 2
    },
    "XRP": {
        "name": "Ripple",
        "description": "Digital payment network and protocol",
        "decimal_places": 4
    },
    "SOL": {
        "name": "Solana",
        "description": "High-performance blockchain platform",
        "decimal_places": 2
    },
    "ADA": {
        "name": "Cardano",
        "description": "Proof-of-stake blockchain platform",
        "decimal_places": 4
    },
    "DOGE": {
        "name": "Dogecoin",
        "description": "Popular meme-based cryptocurrency",
        "decimal_places": 4
    },
    "MATIC": {
        "name": "Polygon",
        "description": "Ethereum scaling solution",
        "decimal_places": 4
    },
    "DOT": {
        "name": "Polkadot",
        "description": "Multi-chain network protocol",
        "decimal_places": 2
    },
    "SHIB": {
        "name": "Shiba Inu",
        "description": "Ethereum-based meme token",
        "decimal_places": 8
    }
}

# Initialize model and tokenizer
MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

class CryptoData(BaseModel):
    symbol: str

def calculate_technical_indicators(df: pd.DataFrame) -> Dict:
    """Calculate various technical indicators from price data."""
    try:
        # Calculate basic metrics
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['EMA'] = df['Close'].ewm(span=20, adjust=False).mean()
        
        # Calculate RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Get latest values
        latest = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else latest
        
        # Convert all numpy values to Python native types
        return {
            'current_price': float(latest['Close']),
            'price_change_24h': float(latest['Close'] - df.iloc[-1440]['Close']) if len(df) >= 1440 else 0.0,
            'price_change_pct': float(((latest['Close'] - previous['Close']) / previous['Close'] * 100)),
            'volume': float(latest['Volume']),
            'ma5': float(latest['MA5']),
            'ma20': float(latest['MA20']),
            'ema': float(latest['EMA']),
            'rsi': float(latest['RSI']),
            'macd': float(latest['MACD']),
            'signal_line': float(latest['Signal_Line']),
            'daily_high': float(df['High'].max()),
            'daily_low': float(df['Low'].min()),
            'daily_volume': float(df['Volume'].sum())
        }
    except Exception as e:
        logger.error(f"Error calculating indicators: {str(e)}")
        raise

def generate_crypto_analysis(data: Dict) -> str:
    """Generate analysis using the SmolLM2 model."""
    try:
        messages = [
            {
                "role": "system", 
                "content": "You are a professional cryptocurrency analyst providing precise technical analysis. You should provide a complete explanation for users and guide them when buy and when sell."
            },
            {
                "role": "user", 
                "content": f"""Analyze these technical indicators for {data['symbol']}:
Current Price: ${data['current_price']:.2f}
24h Change: ${data['price_change_24h']:.2f} ({data['price_change_pct']:.2f}%)
RSI: {data['rsi']:.2f}
5-period MA: ${data['ma5']:.2f}
20-period MA: ${data['ma20']:.2f}
MACD: {data['macd']:.2f}
Signal Line: {data['signal_line']:.2f}
24h High: ${data['daily_high']:.2f}
24h Low: ${data['daily_low']:.2f}
24h Volume: {data['daily_volume']:.2f}

Provide a concise technical analysis focusing on key trends and potential price movement in necessary sentences. provide complete explanation and advice to sell and buy based on technical analysis."""
            }
        ]
        
        input_text = tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
        
        outputs = model.generate(
            inputs,
            max_new_tokens=500,
            temperature=0.2,
            top_p=0.9,
            do_sample=True
        )
        
        # Decode and clean the response
        response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True).strip()
        return response

    except Exception as e:
        logger.error(f"Error generating analysis: {str(e)}")
        raise

@app.get("/analyze/{symbol}")
async def analyze_crypto(symbol: str):
    """Endpoint to analyze a cryptocurrency."""
    try:
        # Validate the symbol
        symbol = symbol.upper()
        if symbol not in SUPPORTED_CRYPTOCURRENCIES:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Unsupported cryptocurrency",
                    "message": f"'{symbol}' is not in the list of supported cryptocurrencies",
                    "supported_cryptos": list(SUPPORTED_CRYPTOCURRENCIES.keys())
                }
            )

        # Fetch cryptocurrency data
        crypto = yf.Ticker(f"{symbol}-USD")
        data = crypto.history(period="1d", interval="1m")
        
        if data.empty:
            raise HTTPException(
                status_code=404, 
                detail={
                    "error": "No data found",
                    "message": f"No price data available for {symbol}",
                    "symbol": symbol
                }
            )
        
        # Calculate indicators
        indicators = calculate_technical_indicators(data)
        indicators['symbol'] = symbol
        indicators['name'] = SUPPORTED_CRYPTOCURRENCIES[symbol]['name']
        
        # Generate analysis
        analysis = generate_crypto_analysis(indicators)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "name": SUPPORTED_CRYPTOCURRENCIES[symbol]['name'],
            "description": SUPPORTED_CRYPTOCURRENCIES[symbol]['description'],
            "indicators": indicators,
            "analysis": analysis
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in analyze_crypto: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail={
                "error": "Internal server error",
                "message": str(e)
            }
        )

@app.get("/supported_cryptos")
async def get_supported_cryptos():
    """Return detailed information about supported cryptocurrencies."""
    return {
        "count": len(SUPPORTED_CRYPTOCURRENCIES),
        "cryptocurrencies": [
            {
                "symbol": symbol,
                "name": info["name"],
                "description": info["description"],
                "decimal_places": info["decimal_places"]
            }
            for symbol, info in SUPPORTED_CRYPTOCURRENCIES.items()
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "device": device,
        "supported_cryptos_count": len(SUPPORTED_CRYPTOCURRENCIES)
    }

@app.get("/author", tags=["About"])
async def get_author_info():
    """Get information about the author of this API."""
    return AUTHOR_INFO

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )

# access to app: http://127.0.0.1:8000/docs