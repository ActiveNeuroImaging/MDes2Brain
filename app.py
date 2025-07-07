import dash
from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import numpy as np
import json
import joblib
import os
import requests
from datetime import datetime

# Russian translations of thought probes
ThoughtProbes = [
    "Я был полностью погружён в свои мысли:",
    "Я думал о других людях:",
    "Я размышлял над проблемами или целями:",
    "В мыслях присутствовали какие-либо звуки или голоса:",
    "Мои мысли содержали звуки:",
    "Мои мысли содержали образы:",
    "Я вспоминал прошлые события:",
    "Мои мысли отвлекали меня от того, что я делал:",
    "Мои мысли были сосредоточены на внешней задаче или деятельности:",
    "Мои мысли были навязчивыми:",
    "Я сознательно направлял и контролировал свои мысли:",
    "Мои мысли были четкими и конкретными:",
    "Я размышлял о будущем:",
    "Мои мысли были окрашены сильными чувствами:",
    "Мои мысли касались меня самого:",
    "Мои мысли содержали информацию, которую я уже знал (например, знания или воспоминания):"
]

# Supabase configuration
SUPABASE_URL = "https://hudbfsdwygfrydpjfnwh.supabase.co"  # Replace with your Supabase URL
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imh1ZGJmc2R3eWdmcnlkcGpmbndoIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTE2NjYyODcsImV4cCI6MjA2NzI0MjI4N30.brUlD4a7dcDuXVwWZsuEkJsn399-mUnvV0JUStGlSX4"  # Replace with your Supabase anon key

def save_to_supabase(data):
    """Save data to Supabase database"""
    url = f"{SUPABASE_URL}/rest/v1/responses"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }
    
    try:
        response = requests.post(url, json=data, headers=headers)
        return response.status_code in [200, 201], response.text
    except Exception as e:
        return False, str(e)

app = Dash(__name__, external_stylesheets=[dbc.themes.SPACELAB])

# Suppress callback exceptions for dynamically generated components
app.config.suppress_callback_exceptions = True

# For deployment
server = app.server

# --- Layouts for each page ---

def page1_layout():
    return dbc.Container([
        html.H2("Информация", className="text-center mb-4"),
        html.P("Добро пожаловать! Это приложение использует генеративную модель, которая собирает субъективные описания вашего опыта и сопоставляет их с фМРТ-данными. На основе этого мы восстанавливаем и визуализируем активность мозга во время изменённого состояния сознания — другими словами, показываем, как мог «выглядеть» ваш мозг, опираясь на то, что вы почувствовали и пережили.",
               className="text-justify mb-4"),
        html.Hr(),
        
        # Participant Information Form
        html.H4("Информация об участнике", className="mb-3"),
        dbc.Row([
            dbc.Col([
                dbc.Label("Возраст:"),
                dbc.Input(id="age-input", type="number", placeholder="Введите ваш возраст", 
                         min=1, max=120, className="mb-3"),
            ], width=12, md=6),
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Label("Пожалуйста, укажите, какое вещество или комбинацию веществ вы приняли:"),
                dbc.Textarea(id="comments-input", placeholder="Укажите одно или более...", 
                           style={"height": "100px"}, className="mb-3"),
            ], width=12),
        ]),
        
        html.Div(id="participant-save-status", className="mb-3"),
        
        # Button container that will be updated based on save status
        html.Div(id="button-container", children=[
            dbc.Button("Сохранить информацию и перейти к вопросам", 
                      id="save-info-btn", color="primary", size="lg", className="w-100 mb-2")
        ]),
    ], className="p-3", fluid=True)

def page2_layout():
    return dbc.Container([
        html.H2("Зондирование мыслей", className="text-center mb-4"),
        dcc.Store(id="question-index", data=0),
        html.Div(id="question-container"),
        
        # Mobile-friendly button layout
        html.Div(id="question-save-status", className="mt-3 mb-3 text-center"),
        
        dbc.Row([
            dbc.Col([
                dbc.Button("Назад", id="prev-btn", color="secondary", size="lg", className="w-100 mb-2")
            ], width=12, md=4),
            dbc.Col([
                dbc.Button("Далее", id="next-btn", color="primary", size="lg", className="w-100 mb-2")
            ], width=12, md=4),
            dbc.Col([
                html.Div(id="brain-button-container")  # Will be populated when data is saved
            ], width=12, md=4),
        ], className="mt-3"),
    ], className="p-3", fluid=True)

def page3_layout():
    return dbc.Container([
        html.H2("Визуализация мозга", className="text-center mb-4"),
        dbc.Row([
            dbc.Col([
                dbc.Label("Показать предсказанную активность мозга:"),
                dbc.Switch(id="brain-switch", value=True, className="mb-3"),
            ], width=12)
        ]),
        
        # Responsive graph container
        html.Div([
            dcc.Graph(id="brain-plot", 
                     style={'width': '100%', 'height': '70vh', 'min-height': '400px'},
                     config={'responsive': True, 'displayModeBar': True})
        ], className="mb-3"),
        
        dbc.Button("Назад к информации", href="/", color="secondary", 
                  size="lg", className="w-100 mb-2"),
    ], className="p-3", fluid=True)


# Main app layout with URL router
app.layout = html.Div([
    # Mobile viewport meta tag
    html.Meta(name="viewport", content="width=device-width, initial-scale=1"),
    dcc.Location(id='url', refresh=False),
    dcc.Store(id='answers', data=[5]*len(ThoughtProbes)),  # Global store for answers
    dcc.Store(id='participant-data', data={}),  # Global store for participant info
    dcc.Store(id='data-saved', data=False),  # Store to track if data has been saved
    html.Div(id='page-content')
])


# Routing callback
@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/questions':
        return page2_layout()
    elif pathname == '/brain':
        return page3_layout()
    else:
        return page1_layout()


# Question display callback
@app.callback(
    Output('question-container', 'children'),
    Output('prev-btn', 'disabled'),
    Output('next-btn', 'children'),
    Input('question-index', 'data'),
    State('answers', 'data'),
)
def display_question(q_index, answers):
    question_text = ThoughtProbes[q_index]
    slider_value = answers[q_index] if answers else 5

    question_card = dbc.Card([
        dbc.CardBody([
            html.H5(f"Вопрос {q_index + 1} из {len(ThoughtProbes)}", className="text-center mb-3"),
            html.P(question_text, style={"fontWeight": "bold"}, className="text-center mb-4"),
            html.Div([
                dcc.Slider(
                    id='current-slider',
                    min=1, max=10, step=1, value=slider_value,
                    marks={1: {"label": "Совсем нет", "style": {"fontSize": "12px"}}, 
                           10: {"label": "Полностью", "style": {"fontSize": "12px"}}},
                    tooltip={"placement": "bottom", "always_visible": True},
                    className="mb-3"
                )
            ], style={"padding": "0 20px"})
        ])
    ], className="mb-3")

    prev_disabled = (q_index == 0)
    next_label = "Завершить" if q_index == len(ThoughtProbes) - 1 else "Далее"

    return question_card, prev_disabled, next_label


# Navigation callback
@app.callback(
    Output('question-index', 'data'),
    Output('question-save-status', 'children'),
    Output('data-saved', 'data'),
    Output('brain-button-container', 'children'),
    Input('next-btn', 'n_clicks'),
    Input('prev-btn', 'n_clicks'),
    State('question-index', 'data'),
    State('answers', 'data'),
    State('participant-data', 'data'),
    State('data-saved', 'data'),
    prevent_initial_call=True
)
def navigate_questions(next_clicks, prev_clicks, question_index, answers, participant_data, data_saved):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    next_clicks = next_clicks or 0
    prev_clicks = prev_clicks or 0
    if answers is None:
        answers = [5] * len(ThoughtProbes)
    if question_index is None:
        question_index = 0

    save_message = ""
    brain_button = ""
    
    if triggered_id == 'next-btn':
        if question_index < len(ThoughtProbes) - 1:
            question_index += 1
        else:
            # Save to Supabase on last question Next click
            try:
                # Generate unique identifier based on timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                
                # Combine all data
                all_data = {
                    "participant_id": timestamp,
                    "timestamp": datetime.now().isoformat(),
                    "age": participant_data.get("age") if participant_data else None,
                    "comments": participant_data.get("comments", "") if participant_data else "",
                    "questionnaire_answers": answers,
                    "questions": ThoughtProbes
                }
                
                success, message = save_to_supabase(all_data)
                if success:
                    save_message = html.Div([
                        html.I(className="fas fa-check-circle me-2"),
                        "Данные успешно сохранены!"
                    ], className="alert alert-success", style={"textAlign": "center"})
                    data_saved = True
                else:
                    save_message = html.Div([
                        html.I(className="fas fa-exclamation-circle me-2"),
                        f"Ошибка сохранения данных"
                    ], className="alert alert-danger", style={"textAlign": "center"})
                    
            except Exception as e:
                save_message = html.Div([
                    html.I(className="fas fa-exclamation-circle me-2"),
                    "Ошибка сохранения данных"
                ], className="alert alert-danger", style={"textAlign": "center"})

    elif triggered_id == 'prev-btn':
        if question_index > 0:
            question_index -= 1

    # Show brain button only if data has been saved
    if data_saved:
        brain_button = dbc.Button("Посмотреть мозг", href="/brain", color="info", 
                                 size="lg", className="w-100 mb-2")

    return question_index, save_message, data_saved, brain_button


# Participant information callback
@app.callback(
    Output('participant-data', 'data'),
    Output('participant-save-status', 'children'),
    Output('url', 'pathname'),
    Output('button-container', 'children'),
    Input('save-info-btn', 'n_clicks'),
    State('age-input', 'value'),
    State('comments-input', 'value'),
    prevent_initial_call=True
)
def save_participant_info(n_clicks, age, comments):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
    
    participant_info = {
        "age": age,
        "comments": comments or "",
        "info_saved_at": datetime.now().isoformat()
    }
    
    success_message = html.Div([
        html.I(className="fas fa-check-circle me-2"),
        "Информация сохранена! Перенаправление к вопросам..."
    ], className="alert alert-success", style={"textAlign": "center"})
    
    # Update button to show it's been saved
    new_button = dbc.Button("Перейти к вопросам", href="/questions", color="success", 
                           size="lg", className="w-100 mb-2")
    
    return participant_info, success_message, "/questions", new_button


# Separate callback to handle slider value updates
@app.callback(
    Output('answers', 'data'),
    Input('current-slider', 'value'),
    State('question-index', 'data'),
    State('answers', 'data'),
    prevent_initial_call=True
)
def update_answer(slider_value, question_index, answers):
    if slider_value is None or question_index is None:
        raise dash.exceptions.PreventUpdate
    
    if answers is None:
        answers = [5] * len(ThoughtProbes)
    
    # Update the current answer
    answers[question_index] = slider_value
    
    return answers


# Brain visualization callback
@app.callback(
    Output('brain-plot', 'figure'),
    Input('brain-switch', 'value'),
    Input('answers', 'data')
)
def update_brain_visualization(brain_switch, answers):
    """
    Update brain visualization based on questionnaire answers and brain switch state.
    """
    try:
        lvert = np.load('lvert.npy', allow_pickle=True)
        gradsRaw = np.load('gradsNormed.npy', allow_pickle=True)
    except FileNotFoundError:
        # Create dummy data if files don't exist
        lvert = np.random.randn(1000, 3)
        gradsRaw = np.random.randn(1000, 5)
    
    n_grad = 5
    grads = np.zeros(gradsRaw.shape)
    gradsMean = grads[:, 0]
    
    if brain_switch and answers and len(answers) == len(ThoughtProbes):
        # Use answers to make predictions
        X = np.array(answers).reshape(1, -1)
        try:
            fitted_models = joblib.load("fitted_models.pkl")
            prediction = np.zeros([n_grad])
            
            for i, model in enumerate(fitted_models):
                prediction[i] = model.predict(X)[0]
            
            for i in range(n_grad):
                grads[:, i] = np.asarray(gradsRaw[:, i]) * prediction[i]
            
            gradsMean = grads.mean(axis=1)
        except Exception as e:
            # If model loading fails, use default visualization
            gradsMean = gradsRaw.mean(axis=1)
    else:
        # Default visualization without prediction
        gradsMean = gradsRaw.mean(axis=1)
    
    x, y, z = lvert[:, 0], lvert[:, 1], lvert[:, 2]
    
    fig = px.scatter_3d(
        x=x,
        y=y,
        z=z,
        color=gradsMean,
        title="Визуализация мозга" + (" (на основе ответов)" if brain_switch and answers else " (по умолчанию)")
    )
    
    
    # Mobile-friendly camera settings
    fig.update_layout(
        scene_camera=dict(
            eye=dict(x=-3.5, y=0, z=0),
            center=dict(x=0, y=0, z=0),
            up=dict(x=0, y=0, z=1)
        ),
        # Better mobile layout
        margin=dict(l=0, r=0, t=40, b=0),
        font=dict(size=4),
        scene=dict(
            #aspectmode='cube',  # Maintain proportions
            dragmode='orbit'    # Better touch interaction
        )
    )
    
    return fig


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    app.run(host='0.0.0.0', port=port, debug=False)
