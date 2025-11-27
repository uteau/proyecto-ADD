
latex_code = df_frecuencias.to_latex(
    float_format="%.2f",
    caption="Frecuencia",
    label="tab:estadisticas",
    escape=False
)

print(latex_code)