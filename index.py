                    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tues June 23 10:40:49 2020

@author: sean
"""

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app
# page4 is buggy
from apps import page1, page2, page3, page5

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content', children=[
        #html.Div(id='hidden-div', style={'display': 'none'})
        dcc.Store(id='store-two-data-upload', storage_type='session')    
        ]),
    html.Div(id='display-page')
])

@app.callback(Output('display-page', 'children'),
            [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/apps/page1':
        return page1.layout
    elif pathname == '/apps/page2':
        return page2.layout
    elif pathname == '/apps/page3':
        return page3.layout
    elif pathname == '/apps/page4':
        return page4.layout
    elif pathname == '/apps/page5':
        return page5.layout
    else:
        return page1.layout

if __name__ == '__main__':
    app.run_server(debug=True)