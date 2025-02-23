from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import ollama
import requests
from typing import List, Dict
from pydantic import BaseModel
import base64
from pathlib import Path
import tempfile
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.requests import Request
import os


app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ProjectDetails(BaseModel):
    construction_type: str
    phase_count: int
    description: str


class ResourcePhase(BaseModel):
    phase_number: int
    materials: Dict[str, Dict[str, float]]
    timeline: str
    dependencies: List[str]


class DocumentStore:
    def __init__(self, embeddings_model):
        self.embeddings_model = embeddings_model
        self.documents = []
        self.embeddings = None

    def add_documents(self, documents):
        self.documents = documents
        texts = [doc.page_content for doc in documents]
        self.embeddings = self.embeddings_model.embed_documents(texts)

    def similarity_search(self, query, k=5):
        query_embedding = self.embeddings_model.embed_query(query)
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return [self.documents[i] for i in top_k_indices]


class IBMGraniteClient:
    def __init__(self, api_key: str, project_id: str):
        self.api_key = api_key
        self.project_id = project_id
        self.auth_token = None

    def get_auth_token(self):
        auth_url = "https://iam.cloud.ibm.com/identity/token"
        auth_data = {
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": self.api_key
        }
        auth_headers = {"Content-Type": "application/x-www-form-urlencoded"}

        response = requests.post(auth_url, data=auth_data, headers=auth_headers)
        self.auth_token = response.json().get("access_token")
        if not self.auth_token:
            raise Exception("Failed to retrieve IBM access token")

    def analyze_phase(self, phase_details: str) -> str:
        if not self.auth_token:
            self.get_auth_token()

        url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.auth_token}"
        }

        prompt = f"""Analyze the following construction phase and provide detailed insights 
        about resource optimization, potential risks, and timeline dependencies:
        {phase_details}"""

        body = {
            "input": prompt,
            "parameters": {
                "decoding_method": "greedy",
                "max_new_tokens": 900,
                "repetition_penalty": 1
            },
            "model_id": "ibm/granite-3-8b-instruct",
            "project_id": self.project_id
        }

        response = requests.post(url, headers=headers, json=body)
        if response.status_code != 200:
            raise Exception(f"IBM Granite API error: {response.text}")

        return response.json().get("results", [{}])[0].get("generated_text", "")


class MaterialInferenceChain:
    def __init__(self, doc_store, granite_client):
        self.doc_store = doc_store
        self.granite_client = granite_client

    async def analyze_image(self, image_path: str, project_details: ProjectDetails) -> List[ResourcePhase]:
        # First get basic material requirements using LLaVA
        llava_response = ollama.chat(
            model="llava:13b",
            messages=[
                {
                    "role": "user",
                    "content": f"""

                   You are an expert construction engineer. Based on the construction drawing image, list every single required construction material with their quantities for a {project_details.construction_type} project.
                    
                    Go through every single item in this list of materials and choose which all are required to construct the above drawing plan:
                    
                    1. Concrete
                    Cement
                    Sand
                    Gravel (Aggregates)
                    Water
                    Reinforcement Steel (Rebars)
                    2. Steel
                    Structural Steel (I-beams, H-beams)
                    Reinforcement Steel (Rebars)
                    Steel Plates
                    Steel Tubes & Pipes
                    3. Wood
                    Softwood (Pine, Cedar)
                    Hardwood (Oak, Maple)
                    Plywood
                    Particle Board
                    Laminated Veneer Lumber (LVL)
                    Timber
                    4. Masonry Materials
                    Clay Bricks
                    Concrete Blocks (CMU)
                    Stone (Granite, Limestone)
                    Mortar
                    5. Glass
                    Float Glass
                    Tempered Glass
                    Laminated Glass
                    Insulated Glass
                    6. Plastics and Polymers
                    Polyvinyl Chloride (PVC)
                    High-Density Polyethylene (HDPE)
                    Polystyrene (PS)
                    Polypropylene (PP)
                    Fiber Reinforced Polymer (FRP)
                    7. Insulation Materials
                    Fiberglass
                    Mineral Wool
                    Polystyrene Foam (EPS, XPS)
                    Polyurethane Foam
                    Insulated Panels
                    8. Roofing Materials
                    Asphalt Shingles
                    Metal Sheets
                    Clay or Concrete Tiles
                    Slate
                    Membrane Roofing (EPDM, TPO)
                    9. Flooring Materials
                    Ceramic Tiles
                    Porcelain Tiles
                    Vinyl Flooring
                    Hardwood Flooring
                    Laminate Flooring
                    Carpet
                    Terrazzo
                    10. Adhesives and Sealants
                    Epoxy Resins
                    Silicone Sealants
                    Acrylic Sealants
                    Polyurethane Sealants
                    11. Paints and Coatings
                    Water-Based Paints
                    Oil-Based Paints
                    Anti-Corrosive Coatings
                    Fire-Resistant Paints
                    12. Plumbing Materials
                    PVC Pipes
                    Copper Pipes
                    PEX Tubing
                    Cast Iron Pipes
                    Faucets, Valves, and Fittings
                    13. Electrical Materials
                    Copper Wiring
                    Conduits (PVC, Metal)
                    Switches & Sockets
                    Electrical Panels
                    Circuit Breakers
                    Light Fixtures
                    14. Fasteners and Connectors
                    Nails
                    Screws
                    Bolts
                    Washers
                    Nuts
                    Rivets
                    15. Doors and Windows
                    Wooden Doors
                    Steel Doors
                    Aluminum Doors & Windows
                    uPVC Doors & Windows
                    16. Gypsum and Plaster
                    Gypsum Boards (Drywall)
                    Plaster of Paris (POP)
                    Cement Plaster
                    Stucco
                    17. Waterproofing Materials
                    Bituminous Waterproofing
                    Liquid Applied Membranes
                    Waterstops
                    Waterproofing Sheets
                    18. Ceramics
                    Ceramic Tiles
                    Porcelain Tiles
                    Sanitary Wares (Sinks, Toilets)
                    19. Metal Products
                    Aluminum Sheets
                    Stainless Steel Sheets
                    Cast Iron Products
                    20. Specialized Materials
                    Geo-synthetics (Geo-textiles, Geo-membranes)
                    Acoustic Panels
                    Fireproofing Materials
                    Solar Panels
                    
                    
                    
                    
                    Provide the materials and quantities in the following format only:
                    material_name: quantity unit
                    
                    Do not include any additional text, explanations, or headers.
                    
                    Example Output:
                    Concrete: 1500 cubic_meters
                    Steel Reinforcement: 250 tons
                    Structural Steel: 800 tons""",
                    "images": [image_path]
                }
            ]
        )

        # Parse LLaVA's material suggestions
        base_materials = self._parse_materials(llava_response["message"]["content"])

        # Generate phases using IBM Granite
        phases = []
        for phase_num in range(1, project_details.phase_count + 1):
            phase_prompt = f"""
            Phase {phase_num} of {project_details.phase_count}
            Project Type: {project_details.construction_type}
            Description: {project_details.description}
            Base Materials: {json.dumps(base_materials, indent=2)}
            """

            phase_analysis = self.granite_client.analyze_phase(phase_prompt)

            # Create phase resource allocation
            phase = ResourcePhase(
                phase_number=phase_num,
                materials=self._allocate_materials(base_materials, phase_num, project_details.phase_count),
                timeline=f"Phase {phase_num} Timeline",
                dependencies=self._extract_dependencies(phase_analysis)
            )
            phases.append(phase)

        return phases

    def _parse_materials(self, llava_output: str) -> Dict[str, Dict[str, float]]:
        # Parse the material list from LLaVA's output
        materials = {}
        for line in llava_output.split('\n'):
            if ':' in line:
                material, quantity_info = line.split(':', 1)
                quantity, unit = quantity_info.strip().split()
                materials[material.strip()] = {
                    "quantity": float(quantity),
                    "unit": unit
                }
        return materials

    def _allocate_materials(self, base_materials: Dict, phase_num: int, total_phases: int) -> Dict:
        # Allocate materials across phases (simplified version)
        phase_materials = {}
        for material, info in base_materials.items():
            phase_materials[material] = {
                "quantity": info["quantity"] / total_phases,
                "unit": info["unit"]
            }
        return phase_materials

    def _extract_dependencies(self, phase_analysis: str) -> List[str]:
        # Extract dependencies from phase analysis (simplified version)
        return ["Previous phase completion"] if phase_analysis else []


# Initialize services
def init_services():
    # Initialize embeddings
    model_name = "BAAI/bge-base-en"
    encode_kwargs = {'normalize_embeddings': True}
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    embeddings_model = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device},
        encode_kwargs=encode_kwargs
    )

    # Initialize document store
    doc_store = DocumentStore(embeddings_model)

    # Initialize IBM Granite client
    granite_client = IBMGraniteClient(
        api_key="9FV7l0Jxqe7ceL09MeH_g9bYioIQuABXsr1j1VHbKOpr",
        project_id="4aa39c25-19d7-48c1-9cf6-e31b5c223a1f"
    )

    return doc_store, granite_client


# Initialize services
doc_store, granite_client = init_services()

# Initialize inference chain
inference_chain = MaterialInferenceChain(doc_store, granite_client)


# Initialize Jinja2 templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze-project/")
async def analyze_project(
        project_details: ProjectDetails,
        file: UploadFile = File(...),
):
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()

            # Analyze the project
            phases = await inference_chain.analyze_image(
                temp_file.name,
                project_details
            )

            return {
                "project_type": project_details.construction_type,
                "phases": phases
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

import uvicorn

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)