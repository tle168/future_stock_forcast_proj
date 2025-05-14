import pandas as pd

def export_excel(df_dict):
    with pd.ExcelWriter("kq_loc_co_phieu.xlsx") as writer:
        for code, df in df_dict.items():
            if not df.empty:
                df.to_excel(writer, sheet_name=code[:31], index=False)
