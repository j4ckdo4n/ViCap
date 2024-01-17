from pyngrok import ngrok
...
url = ngrok.connect(5000)
print('Henzy Tunnel URL:', url)