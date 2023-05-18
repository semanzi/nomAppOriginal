#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tues June 23 10:40:49 2020

@author: sean
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
import networkx as nx
import matplotlib as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from app import app

layout = html.Div(children=[
    html.Div(id='home-p5', className='padheader color-section',children=[
        html.Div(className='container',children=[
            html.Div(className='row',children=[
                html.Div(className='col-4 logo',children=[
                    html.A(href='/apps/page1', children=[
                    html.Img(src=app.get_asset_url('images/project_logo_v8.png')),
                    ]),
                ]),
                html.Div(className='col-8 text-center site-title',children=[
                    html.Div(className='site-title-container',children=[
                        html.H1(className='flex-header',children=['Network-based operational modelling platform']),
                    ]),
                ]),
            ]),
        ]),
    ]),
    html.Div(id='navbar-p5', children=[
        html.Div(className='padheader color-section', children=[
            html.Div(className='row', children=[
                html.Div(className='container', children=[
                    html.Ul(id='nav-p5', className='nav', children=[
                        html.Li(children=[html.A(children=['Instructions'],href="/apps/page1")]),
                        html.Li(children=[html.A(children=['Data upload'], href="/apps/page2")]),
                        html.Li(children=[html.A(children=['Network overview'], href="/apps/page3")]),
                        html.Li(children=[html.A(children=['Dynamic network'], href="/apps/page4")]),
                        html.Li(children=[html.A(className='active', children=['Reports'], href="/apps/page5")]),
                    ]),
                ]),
            ]),
        ]),
    ]),
    html.Div(id='page-title-p5', className='padheader color-section', children=[
        html.Div(className='row', children=[
            html.Div(className='container', children=[
                html.H2(className='text-center', children=['Reports']),
            ]),
        ]),
    ]),
    html.Div(className='padsection', children=[
        html.Div(className='row', children=[
            html.Div(className='b-container', children=[
                html.H3(children=['Select report']),
                html.P(children=['Some text here']),
            ]),
        ]),
    ]),
    html.Div(id='contact-p5', className='padsection footer-border', children=[
        html.Div(className='b-container', children=[
            html.Div(className='row', children=[
                html.Div(className='col-2', children=[
                    html.H3(children=['Contact details']),
                    html.Div(className='details-container', children=[
                        html.Ul(className='list-unstyled', children=[
                            html.Li(children=[html.Span(children=['Dr Sean Manzi'])]),
                            html.Li(children=[html.Span(children=['South Cloisters'])]),
                            html.Li(children=[html.Span(children=['St Lukes campus'])]),
                            html.Li(children=[html.Span(children=['University of Exeter'])]),
                            html.Li(children=[html.Span(children=['Exeter, Devon'])]),
                            html.Li(children=[html.Span(children=['EX1 2LU'])]),
                            html.Li(children=[html.Span(children=['+44(0)1392 726096'])]),
                        ]),
                    ]),
                ]),
                html.Div(className='col-2', children=[
                    html.Div(className='email-container', children=[
                        html.P(children=['Send an email with any questions or comments to Dr Sean Manzi at:']),
                        html.A(className='contact-email', children=['s.s.manzi@exeter.ac.uk'], href='mailto:s.s.manzi@exeter.ac.uk'),
                    ]),
                ]),
                html.Div(className='col-8', children=[
                    html.Div(className='funding-container text-center', children=[
                        html.P(children=['This project is funded and supported by:']),
                        html.Ul(className='list-unstyled funding', children=[
                            html.Li(children=[html.A(children=[html.Img(src=app.get_asset_url('images/THIS_logo2.png'))], href="https://www.thisinstitute.cam.ac.uk/", target='_blank')]),
                            html.Li(children=[html.A(children=[html.Img(src=app.get_asset_url('images/Exeter_Uni.png'))], href="http://www.exeter.ac.uk/", target='_blank')]),
                            html.Li(children=[html.A(children=[html.Img(src=app.get_asset_url('images/NIHR_Logo.png'))], href="https://www.arc-swp.nihr.ac.uk/", target='_blank')]),
                        ]),
                    ]),
                ]),
            ]),
        ]),
    ]),
    html.Div(id='licence-p5', className='padsection color-section footer-border', children=[
        html.Div(className='row', children=[
            html.Div(className='b-container text-center', children=[
                html.P(children=['Copyright 2020 Sean Manzi']),
                html.P(children=['Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:']),
                html.P(children=['The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.']),
                html.P(children=['THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.']),
            ]),
        ]),
    ]),
])