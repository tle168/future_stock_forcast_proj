import plotly.graph_objs as go
import plotly.express as px

def plot_price_forecast(data, forecast_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Thực tế',line=dict(color='green')))
    fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Forecast'], name='Dự báo',line=dict(color='red')))
    fig.update_layout(title='Dự báo giá cổ phiếu', xaxis_title='Ngày', yaxis_title='Giá')
    return fig

def plot_heatmap(data):
    df = data.copy()
    df['Date'] = df.index
    df['Day'] = df['Date'].dt.day.astype(str)
    df['Month'] = df['Date'].dt.month.astype(str)
    pivot = df.pivot_table(index='Month', columns='Day', values='Volume', aggfunc='mean')
    fig = px.imshow(pivot, color_continuous_scale='Reds', aspect='auto', labels=dict(color='Volume'))
    fig.update_layout(title='Heatmap khối lượng giao dịch theo ngày')
    return fig
def plot_price_forecast(df):
    # Example implementation for plotting price forecast
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='Close Price'))
    return fig

def plot_heatmap(data):
    # Example implementation for plotting a heatmap
    fig = go.Figure(data=go.Heatmap(z=data))
    return fig

def plot_recommendation_chart(df):
    # Example implementation for plotting recommendation chart
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df['recommendation'], name='Recommendation'))
    return fig