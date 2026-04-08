import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ─── Healthcare Knowledge Base ───────────────────────────────────────────────
# 20 short, factual healthcare knowledge statements.
# These are educational / preventive — no diagnosis or prescriptions.
DOCUMENTS = [
    # Hypertension
    "Hypertension is defined as a blood pressure reading consistently above 140/90 mmHg.",
    "Common risk factors for hypertension include obesity, high salt intake, smoking, physical inactivity, and chronic stress.",
    "Hypertension is often called the 'silent killer' because it typically has no obvious symptoms until it causes organ damage.",
    "Lifestyle modifications for hypertension include reducing sodium intake, increasing potassium-rich foods, exercising regularly, and limiting alcohol.",

    # Type 2 Diabetes
    "Type 2 diabetes occurs when the body becomes resistant to insulin or does not produce enough insulin to maintain normal blood glucose levels.",
    "Key risk factors for type 2 diabetes include obesity, sedentary lifestyle, family history, age above 45, and high blood pressure.",
    "Symptoms of type 2 diabetes include frequent urination, increased thirst, fatigue, blurred vision, and slow-healing wounds.",
    "Preventive strategies for type 2 diabetes include maintaining a healthy weight, eating a balanced diet, and getting at least 150 minutes of moderate exercise per week.",

    # Anemia
    "Anemia is a condition in which the blood lacks enough healthy red blood cells or hemoglobin to carry adequate oxygen to the body's tissues.",
    "Common symptoms of anemia include fatigue, weakness, pale or yellowish skin, irregular heartbeat, shortness of breath, and dizziness.",
    "Iron-deficiency anemia is the most common form and is caused by insufficient dietary iron, poor iron absorption, or blood loss.",
    "Foods rich in iron include red meat, beans, lentils, spinach, fortified cereals, and dried apricots.",

    # Cardiovascular Disease
    "Cardiovascular disease (CVD) refers to conditions affecting the heart and blood vessels, including coronary artery disease, heart failure, and stroke.",
    "Major risk factors for cardiovascular disease include high blood pressure, high LDL cholesterol, smoking, diabetes, obesity, and physical inactivity.",
    "Regular aerobic exercise for at least 30 minutes on most days of the week significantly reduces the risk of cardiovascular disease.",
    "A heart-healthy diet is rich in fruits, vegetables, whole grains, lean proteins, and healthy fats while low in saturated fats, trans fats, and sodium.",

    # Obesity
    "Obesity is defined as a Body Mass Index (BMI) of 30 or higher and is associated with increased risk of type 2 diabetes, heart disease, and certain cancers.",
    "Causes of obesity include excessive caloric intake, sedentary lifestyle, genetic predisposition, hormonal imbalances, and certain medications.",

    # Nutrition & Preventive Healthcare
    "Adequate hydration — typically 8 glasses of water per day — supports kidney function, digestion, and temperature regulation.",
    "Preventive healthcare recommendations include regular blood pressure checks, blood glucose screening, cholesterol testing, vaccinations, and annual physical exams.",
]

# ─── ChromaDB Initialization ─────────────────────────────────────────────────
def init_database():
    """Initialises or loads the ChromaDB vector store with healthcare knowledge."""
    persist_directory = "./chroma_db_health"

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if not os.path.exists(persist_directory):
        print("Initialising Healthcare ChromaDB with sample facts...")
        vectorstore = Chroma.from_texts(
            texts=DOCUMENTS,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name="healthcare_knowledge"
        )
    else:
        print("Loading existing Healthcare ChromaDB...")
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            collection_name="healthcare_knowledge"
        )

    return vectorstore


def get_retriever(k: int = 4):
    """Returns a retriever for the healthcare vector store."""
    vectorstore = init_database()
    return vectorstore.as_retriever(search_kwargs={"k": k})


def retrieve_facts(query: str, k: int = 4) -> str:
    """Retrieves top-k relevant healthcare facts for a given query."""
    retriever = get_retriever(k=k)
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant facts found in the knowledge base."
    return "\n".join([f"- {doc.page_content}" for doc in docs])
