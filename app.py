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
    "Я был поглощен содержанием своих мыслей:",
    "Мои мысли касались других людей:",
    "Я думал о решениях проблем (или целей):",
    "Мои мысли содержали слова:",
    "Мои мысли содержали звуки:",
    "Мои мысли содержали образы:",
    "Мои мысли касались прошлых событий:",
    "Мои мысли отвлекали меня от того, что я делал:",
    "Мои мысли были сосредоточены на внешней задаче или деятельности:",
    "Мои мысли были навязчивыми:",
    "Мои мысли были преднамеренными:",
    "Мои мысли были подробными и конкретными:",
    "Мои мысли касались будущих событий:",
    "Эмоция моих мыслей была:",
    "Мои мысли касались меня самого:",
    "Мои мысли содержали информацию, которую я уже знал (например, знания или воспоминания):"
]

# Supabase configuration
SUPABASE_URL = "https://your-project.supabase.co"  # Replace with your Supabase URL
SUPABASE_KEY = "your-anon-key"  # Replace with your Supabase anon key

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
        html.H2("Информация"),
        html.P("Добро пожаловать! Это приложение связывает выборочное исследование опыта и активность мозга."),
        html.Hr(),
        
        # Participant Information Form
        html.H4("Информация об участнике"),
        dbc.Row([
            dbc.Col([
                dbc.Label("Возраст:"),
                dbc.Input(id="age-input", type="number", placeholder="Введите ваш возраст", min=1, max=120),
            ], width=6),
        ], className="mb-3"),
        
        dbc.Row([
            dbc.Col([
                dbc.Label("Комментарии:"),
                dbc.Textarea(id="comments-input", placeholder="Любые дополнительные комментарии или заметки...", 
                           style={"height": "100px"}),
            ], width=12),
        ], className="mb-3"),
        
        html.Div(id="participant-save-status", style={"color": "green", "margin": "10px 0"}),
        
        dbc.Row([
            dbc.Col(dbc.Button("Сохранить информацию и перейти к вопросам", id="save-info-btn", color="primary"), width="auto"),
            dbc.Col(dbc.Button("Посмотреть мозг", href="/brain", color="info"), width="auto"),
        ], className="gap-2"),
    ], className="p-4")

def page2_layout():
    # Hidden stores for question index
    return dbc.Container([
        html.H2("Зондирование мыслей"),
        dcc.Store(id="question-index", data=0),
        html.Div(id="question-container"),
        dbc.Row([
            dbc.Col(dbc.Button("Назад", id="prev-btn", color="secondary"), width="auto"),
            dbc.Col(dbc.Button("Далее", id="next-btn", color="primary"), width="auto"),
            dbc.Col(dbc.Button("Посмотреть мозг", href="/brain", color="info"), width="auto"),
            dbc.Col(html.Div(id="save-status", style={"paddingLeft": "20px", "color": "green"}))
        ], className="mt-3 align-items-center"),
    ], className="p-4")

def page3_layout():
    # Brain visualization page with dynamic updates based on answers
    return dbc.Container([
        html.H2("Визуализация мозга"),
        dbc.Row([
            dbc.Col([
                dbc.Label("Использовать предсказание мозга:"),
                dbc.Switch(id="brain-switch", value=False, className="mb-3"),
            ], width=12)
        ]),
        dcc.Graph(id="brain-plot", style={'width': '100vh', 'height': '100vh'}),
        dbc.Button("Назад к информации", href="/", color="secondary", className="mt-3"),
    ], className="p-4")


# Main app layout with URL router
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dcc.Store(id='answers', data=[5]*len(ThoughtProbes)),  # Global store for answers
    dcc.Store(id='participant-data', data={}),  # Global store for participant info
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
            html.H5(f"Вопрос {q_index + 1} из {len(ThoughtProbes)}"),
            html.P(question_text, style={"fontWeight": "bold"}),
            dcc.Slider(
                id='current-slider',
                min=1, max=10, step=1, value=slider_value,
                marks={1: "Совсем нет", 10: "Полностью"},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ])
    ])

    prev_disabled = (q_index == 0)
    next_label = "Завершить" if q_index == len(ThoughtProbes) - 1 else "Далее"

    return question_card, prev_disabled, next_label


# Navigation callback (without slider input)
@app.callback(
    Output('question-index', 'data'),
    Output('save-status', 'children'),
    Input('next-btn', 'n_clicks'),
    Input('prev-btn', 'n_clicks'),
    State('question-index', 'data'),
    State('answers', 'data'),
    State('participant-data', 'data'),
    prevent_initial_call=True
)
def navigate_questions(next_clicks, prev_clicks, question_index, answers, participant_data):
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
                    save_message = f"Данные успешно сохранены! ID участника: {timestamp}"
                else:
                    save_message = f"Ошибка сохранения данных: {message}"
                    
            except Exception as e:
                save_message = f"Ошибка сохранения данных: {e}"

    elif triggered_id == 'prev-btn':
        if question_index > 0:
            question_index -= 1

    return question_index, save_message


# Participant information callback
@app.callback(
    Output('participant-data', 'data'),
    Output('participant-save-status', 'children'),
    Output('url', 'pathname'),
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
    
    return participant_info, "Информация сохранена! Перенаправление к вопросам...", "/questions"


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
    Input('answers', 'data')  # Changed from State to Input to trigger updates
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
    
    # Debug info - you can remove this later
    print(f"Brain switch: {brain_switch}")
    print(f"Answers: {answers}")
    print(f"Answers length: {len(answers) if answers else 'None'}")
    
    if brain_switch and answers and len(answers) == len(ThoughtProbes):
        # Use answers to make predictions
        X = np.array(answers).reshape(1, -1)
        try:
            fitted_models = joblib.load("fitted_models.pkl")
            prediction = np.zeros([n_grad])
            
            for i, model in enumerate(fitted_models):
                prediction[i] = model.predict(X)[0]
            
            print(f"Predictions: {prediction}")  # Debug info
            
            for i in range(n_grad):
                grads[:, i] = np.asarray(gradsRaw[:, i]) * prediction[i]
            
            gradsMean = grads.mean(axis=1)
            print("Using prediction-based visualization")  # Debug info
        except Exception as e:
            print(f"Model loading error: {e}")  # Debug info
            # If model loading fails, use default visualization
            gradsMean = gradsRaw.mean(axis=1)
    else:
        # Default visualization without prediction
        gradsMean = gradsRaw.mean(axis=1)
        print("Using default visualization")  # Debug info
    
    x, y, z = lvert[:, 0], lvert[:, 1], lvert[:, 2]
    
    fig = px.scatter_3d(
        x=x,
        y=y,
        z=z,
        color=gradsMean,
        title="Визуализация мозга" + (" (на основе ответов)" if brain_switch and answers else " (по умолчанию)")
    )
    
    # Set camera viewing angle
    fig.update_layout(
        scene_camera=dict(
            eye=dict(x=-2, y=0, z=0),  # Camera position
            center=dict(x=0, y=0, z=0),     # Point camera looks at
            up=dict(x=0, y=0, z=1)          # Up direction
        )
    )
    
    return fig


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    app.run(host='0.0.0.0', port=port, debug=False)
