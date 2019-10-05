import pandas as pd

def treat_economatica_indicadores_financeiros(csv_file):
    # Read CSV file
    df = pd.read_csv(csv_file).transpose()
    # Set first row as column name
    df.columns = df.iloc[0]
    # Remove row that had column names
    df = df.iloc[1:, ]
    # Convert all row values from type object to float
    for col in df.columns:
        df[col] = df[col].astype(float)
    # Prepare data
    df = prepare_data(df)
    # Add column "Data" with date time value
    df['Data'] = pd.to_datetime(df.index.values)
    # Set index as date time index
    df = df.set_index('Data')
    return df


def prepare_data(data):
    # Interpolate missing values
    df = data.interpolate(method='linear', axis=0)
    # Drop rows if there are still NaN values
    df.dropna(inplace=True)
    return df

def treat_economatica_stock_with_following_month_opening_price(csv_file):
    # Read csv file containing stock data
    df = pd.read_csv(csv_file)
    # Set column 'Data' as datetime
    df['Data'] = pd.to_datetime(df['Data'])
    # Set index as 'Data'
    df = df.set_index('Data')
    # Group data by quarter and get first value of quarter grouping
    quarter = df.resample('Q').first()
    # Create new column with the information of the following month
    quarter['Abertura proximo mes'] = quarter.Abertura.shift(-1)
    return quarter
