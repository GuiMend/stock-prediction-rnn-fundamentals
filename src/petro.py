# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import tratamento_dados_empresa

X = tratamento_dados_empresa.treat_economatica_indicadores_financeiros(
        '../data/petrobras-indicadores-financeiros-raw.csv')

y = tratamento_dados_empresa.treat_economatica_stock_with_following_month_opening_price(
        '../data/petrobras-cotacao-raw.csv')

y = y.iloc[len(y) - len(X) - 1: len(y) - 1,:]