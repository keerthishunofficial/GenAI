import pandas as pd
from app import build_graph

def test():
    df = pd.read_csv("sample_data.csv")
    graph = build_graph()
    state = {
        "raw_df": df,
        "clean_df": pd.DataFrame(),
        "report": {}
    }
    
    print("Running pipeline...")
    final_state = graph.invoke(state)
    
    print("Report:", final_state["report"])
    print("Clean DF rows:", len(final_state["clean_df"]))
    print("Success!")

if __name__ == "__main__":
    test()
