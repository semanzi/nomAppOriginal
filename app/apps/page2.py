#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tues June 23 10:40:49 2020

@author: sean
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import networkx as nx
import matplotlib as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import json
import base64
import io
from dash.exceptions import PreventUpdate

from main import dapp as app

layout = html.Div(children=[
    html.Div(id='home-p2', className='padheader color-section',children=[
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
    html.Div(id='navbar-p2', children=[
        html.Div(className='padheader color-section', children=[
            html.Div(className='row', children=[
                html.Div(className='container', children=[
                    html.Ul(id='nav-p2', className='nav', children=[
                        html.Li(children=[html.A(children=['Instructions'],href="/apps/page1")]),
                        html.Li(children=[html.A(className='active', children=['Data upload'], href="/apps/page2")]),
                        html.Li(children=[html.A(children=['Network overview'], href="/apps/page3")]),
                        html.Li(children=[html.A(children=['Dynamic network'], href="/apps/page4")]),
                        html.Li(children=[html.A(children=['Reports'], href="/apps/page5")]),
                    ]),
                ]),
            ]),
        ]),
    ]),
    html.Div(id='page-title-p2', className='padheader color-section', children=[
        html.Div(className='row', children=[
            html.Div(className='container', children=[
                html.H2(className='text-center', children=['Data upload']),
            ]),
        ]),
    ]),
    html.Div(id='upload-instruc', className='padsection', children=[
        html.Div(className='row', children=[
            html.Div(className='b-container', children=[
                html.H3(children=['Upload instructions']),
                html.P(children=['Some text here']),
            ]),
        ]),
    ]),
    html.Div(id='upload', className='padsection color-section', children=[
        html.Div(className='row', children=[
            html.Div(className='b-container', children=[
                html.H3(children=['File upload']),
                html.Div(className='text-center', children=[
                    dcc.Upload(id='upload-data', className='upload-obj obj-center', children=[
                        html.Div(children=['Drag and Drop or ', html.A('Select Files')])
                    ]),
#                    html.Div(id='output-data-upload'),
                    dcc.Store(id='store-data-upload', storage_type='session'),
                ]),
            ]),
        ]),
    ]),
    html.Div(id='errors', className='padsection', children=[
        html.Div(className='row', children=[
            html.Div(className='b-container', children=[
                html.H3(children=['Upload and parsing errors']),
                html.P(children=['Some text here']),
                html.Div(className='row', children=[
                    html.Div(className='col-6', children=[
                        html.H4(children=['Upload errors']),
                        html.Ul(id='upload-error-list')
                    ]),
                    html.Div(className='col-6', children=[
                        html.H4(children=['Parsing errors']),
                        html.Ul(id='parsing-error-list')
                    ]),
                ]),
            ]),
        ]),
    ]),
    html.Div(id='upload-summary', className='padsection color-section', children=[
        html.Div(className='row', children=[
            html.Div(className='b-container', children=[
                html.H3(children=['Uploaded data summary']),
                html.P(children=['Some text here']),
                html.Div(id='summary_table_container'),
            ]),
        ]),
    ]),
    html.Div(id='clean-data', className='padsection', children=[
        html.Div(className='row', children=[
            html.Div(className='b-container', children=[
                html.H3(children=['Data cleaning tools']),
                html.P(children=['Some text here']),
                html.Div(className='b-container', children=[
                    dcc.Checklist(id='clean-checklist',
                        options=[
                            {'label': 'Remove excess whitespace', 'value': 1},
                            {'label': 'Convert text to lowercase', 'value': 2}
                            ]),
                        html.Button('Clean data', id='clean-data', n_clicks=0, disabled=True),
                    ]),
                #dcc.Store(id='store-two-data-upload', storage_type='session'),
                html.Div(id='clean-actions'),
            ]),
        ]),
    ]),
    html.Div(id='remove-entries', className='padsection color-section', children=[
        html.Div(className='row', children=[
            html.Div(className='b-container', children=[
                html.H3(children=['Entries flagged for removal']),
                html.P(children=['Some text here']),
                html.Div(id='removal-checklist-container'),
            ]),
        ]),
    ]),
    html.Div(id='contact-p2', className='padsection footer-border', children=[
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
    html.Div(id='licence-p2', className='padsection color-section footer-border', children=[
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

def import_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return None
    return df

def check_columns(df):
    names = list(df.columns)
    expected = ['ID', 'ReferralDate', 'DischargeDate', 'Team']
    missing = []
    for i in range(len(expected)):
        if expected[i] != names[i]:
            missing.append(i)
    if missing != []:
        return [html.Li(expected[i] + ' was not found in column ' + str(i)) for i in missing]
    else:
        return [html.Li('All expected columns are present and in the correct position')]

def check_format(df):
    df['ReferralDate'] = pd.to_datetime(df['ReferralDate'], format='%d/%m/%Y')
    df['DischargeDate'] = pd.to_datetime(df['DischargeDate'], format='%d/%m/%Y')
    #dataTypes = dict(df.dtypes)
    format_issues = []
    if df.dtypes['ID'] != np.int64:
        format_issues.append(html.Li('Error ID column not correct format; check if numerical'))
    if pd.core.dtypes.common.is_datetime64_ns_dtype(df['ReferralDate']) == False:
        format_issues.append(html.Li('Error ReferralDate column not correct format; check if datetime in the format day/month/Year e.g. 1/4/2020'))
    if pd.core.dtypes.common.is_datetime64_ns_dtype(df['DischargeDate']) == False:
        format_issues.append(html.Li('Error DischargeDate column not correct format; check if datetime in the format day/month/Year e.g. 1/4/2020'))
    if df.dtypes['Team'] != np.object:
        format_issues.append(html.Li('Error Team column not correct format; check if a text string'))
    if format_issues == []:
        format_issues.append(html.Li('All required columns are in the correct format'))
    return format_issues

def summary_table(df):
    names = list(df.columns)
    n_entries = pd.DataFrame(df.count())
    p_complete = round((n_entries / len(df)) * 100, 0)
    t_summary = pd.DataFrame({'Column name': names, 'Number of entries': n_entries[0], 'Percentage completeness': p_complete[0]})
    upload_table = dbc.Table.from_dataframe(t_summary)
    return upload_table

def cat_removal(df):
    names = list(df.columns)
    check_list = []
    for i in range(3,len(df.columns)):
        a = np.array(df.iloc[:,i])
        unique, counts = np.unique(a, return_counts=True)
        count_dict = dict(zip(unique, counts))
        low_dict = {k:v for (k,v) in count_dict.items() if v <= 1}
        key = list(low_dict.keys())
        check_list.append([{'label':'Column: ' + names[i] + ', Category: ' + key[k], 'value':key[k]} for k in range(len(key))])
    flat_list = [item for sublist in check_list for item in sublist]
    checklist_obj = dcc.Checklist(options=flat_list)
    return checklist_obj

def remove_whitespace(df):
    names = list(df.columns)
    for i in names:
        if df[i].dtype == np.object:
            df[i] = df[i].str.strip()
    return df

def to_lowercase(df):
    names = list(df.columns)
    for i in names:
        if df[i].dtype == np.object:
            df[i] = df[i].str.lower()
    return df

@app.callback([Output('store-data-upload', 'data'),
            Output('upload-error-list', 'children'),
            Output('parsing-error-list', 'children'),
            Output('summary_table_container', 'children'),
            #Output('removal-checklist-container', 'children'),
            Output('clean-data', 'disabled')],
            [Input('upload-data', 'contents')],
            [State('upload-data', 'filename')])
# @app.callback(Output('upload-error-list', 'children'),
#               [Input('upload-data', 'contents')],
#               [State('upload-data', 'filename')])
def upload_data(contents, filename):
    if contents is not None:
        children = import_contents(contents, filename)
        col_errors = check_columns(children)
        format_errors = check_format(children)
        if children is not None:
            upload_table = summary_table(children)
            #check_list = cat_removal(children)
            children['ReferralDate'] = children['ReferralDate'].astype(str)
            children['DischargeDate'] = children['DischargeDate'].astype(str)
            children = children.to_dict()
            children = json.dumps(children)
            activate_btns = False
            #return children, col_errors, format_errors, upload_table, check_list, activate_btns
            return children, col_errors, format_errors, upload_table, activate_btns
        # else:
        #     upload_error = [html.Li('Error processing file, ensure CSV format')]
        #     return upload_error
        #    return html.Li('Done')
        
@app.callback(Output('store-two-data-upload', 'data'),
              [Input('clean-data', 'n_clicks'),
                Input('clean-checklist', 'value'),
                Input('store-data-upload', 'data')])
def clean_data(btn1, vals, data):
    if vals is not None:
        if len(vals) > 0 and btn1 != 0:
            data = json.loads(data)
            selection = vals
            #selected = list(selection.values())
            total = sum(selection)
            df = pd.DataFrame.from_dict(data)
            if total == 1:
                clean_df = remove_whitespace(df)
                clean_dict = clean_df.to_dict()
                data_dict = json.dumps(clean_dict)
                return data_dict
            elif total == 2:
                clean_df = to_lowercase(df)
                clean_dict = clean_df.to_dict()
                data_dict = json.dumps(clean_dict)
                return data_dict
            elif total == 3:
                clean_df = remove_whitespace(df)
                clean_df = to_lowercase(clean_df)
                clean_dict = clean_df.to_dict()
                data_dict = json.dumps(clean_dict)
                return data_dict

@app.callback(Output('clean-actions', 'children'),
              [Input('clean-data', 'n_clicks'),
                Input('clean-checklist', 'value')])
def clean_actions_list(btn1, vals):
    if vals is not None:
        if len(vals) > 0 and btn1 != 0:
            selection = vals
            #selected = list(selection.values())
            total = sum(selection)
            if total == 1:
                list_item = html.Li('Leading and trailing whitespace removed from the dataset')
                return list_item
            elif total == 2:
                list_item = html.Li('All text has been converted to lowercase')
                return list_item
            elif total == 3:
                list_item = [html.Li('Leading and trailing whitespace removed from the dataset'),
                              html.Li('All text has been converted to lowercase')]
                return list_item
        