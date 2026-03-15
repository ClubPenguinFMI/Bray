import yfinance as yf

def get_ticker(text):
    search = yf.Search(query=text, max_results=1).search()
    ls = list(search._all.values())[0]
    if len(ls) == 0:
        return None
    return ls[0]

def main():
    company_name = "SK Hynix"
    ticker = get_ticker(company_name)
    print(f"The ticker for {company_name} is {ticker}.")


if __name__ == "__main__":
    main()