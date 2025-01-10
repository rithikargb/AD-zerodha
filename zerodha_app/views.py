from django.shortcuts import render,redirect
from django.http import JsonResponse,HttpResponse
import yfinance as yf
from openai import OpenAI
from . import models


def generate_stock_graph(request,symbol,start='2024-08-30',end='2024-12-30'):

    data = models.stock_data.objects.filter(Ticker = symbol, Date__range = [start,end])

    open_values = [entry.Open for entry in data]
    high_values = [entry.High for entry in data]
    low_values = [entry.Low for entry in data]
    close_values = [entry.Close for entry in data]
    dates = [entry.Date.strftime('%Y-%m-%d') for entry in data]

    fig_data = [{
        'x': dates,
        'open': open_values,
        'high': high_values,
        'low': low_values,
        'close': close_values,
    }]

    return JsonResponse({'data': fig_data})



def chatBot(request,question):
    client = OpenAI(api_key="YOUR_API_KEY")
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role":"system",
                    'content':'You are a helpful assistant. Answer about stocks and their symbols.Suggest symbols other than these - sensex,nifty,AAPL,AMZN,RELI,INTC,NVDA,F,TSLA,META,MSFT,GOOG'
                },
                {
                    'role':'user',
                    'content':question
                },
            ]
        )
        response = completion.choices[0].message.content
        return JsonResponse({'answer':response})
    except Exception as e:
        return JsonResponse({'error':str(e)},status=500)



def getData(sym):
    data = yf.Ticker(sym).history(period="5d")
    cur = data.iloc[-1]
    prev = data.iloc[-2]

    return {'val':cur['Open'],
            'diff':cur['Open']-prev['Close']}





def home(request):
    try:
        stockSNX = yf.Ticker('^BSESN')
        dataSNX = stockSNX.history(period="5d")
        stockNFT = yf.Ticker('^NSEI')
        dataNFT  = stockNFT.history(period="5d")

        if len(dataSNX) < 2 or len(dataNFT) < 2:
            if len(dataSNX) == 0 or len(dataNFT) == 0:
                return JsonResponse({'error': 'Market is currently closed, and no data is available.'}, status=400)
            return JsonResponse({'error': 'Not enough data for the last 2 days'}, status=400)
        
        latest_dataSNX = dataSNX.iloc[-1]
        previous_dataSNX = dataSNX.iloc[-2]

        latest_dataNFT = dataNFT.iloc[-1]
        previous_dataNFT = dataNFT.iloc[-2]

        sensex_ = latest_dataSNX['Open']
        nifty_ = latest_dataNFT['Close']
        sensex_diff = latest_dataSNX['Open']-previous_dataSNX['Close']
        nifty_diff = latest_dataNFT['Open']-previous_dataNFT['Close']

        AAPLdata = getData('AAPL')
        AMZNdata = getData('AMZN')
        NFLXdata = getData('NFLX')
        BRKBdata = getData('BRK-B')
        NVDAdata = getData('NVDA')
        JPMdata = getData('JPM')
        TSLAdata = getData('TSLA')
        METAdata = getData('META')
        MSFTdata = getData('MSFT')
        GOOGdata = getData('GOOG')

        sensex_data = {
            'sensex': sensex_,
            'nifty' : nifty_,
            'sensex_diff':sensex_diff,
            'nifty_diff':nifty_diff,
            'AAPL':AAPLdata['val'],
            'AMZN':AMZNdata['val'],
            'BRKB':BRKBdata['val'],
            'NFLX':NFLXdata['val'],
            'NVDA':NVDAdata['val'],
            'JPM':JPMdata['val'],
            'TSLA':TSLAdata['val'],
            'META':METAdata['val'],
            'MSFT':MSFTdata['val'],
            'GOOG':GOOGdata['val'],
            'AAPLdiff':AAPLdata['diff'],
            'AMZNdiff':AMZNdata['diff'],
            'NFLXdiff':NFLXdata['diff'],
            'BRKBdiff':BRKBdata['diff'],
            'NVDAdiff':NVDAdata['diff'],
            'JPMdiff':JPMdata['diff'],
            'TSLAdiff':TSLAdata['diff'],
            'METAdiff':METAdata['diff'],
            'MSFTdiff':MSFTdata['diff'],
            'GOOGdiff':GOOGdata['diff']
        }
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

    # Render the data to the template
    return render(request, 'homeui.html', sensex_data)