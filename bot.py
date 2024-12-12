import discord
from discord import app_commands
import numpy as np
from fake_useragent import UserAgent
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from discord.ext import commands
import tls_client
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import os

intents = discord.Intents.default()
intents.messages = True
intents.members = True
client = commands.Bot(command_prefix="!", intents=intents)

@client.event
async def on_ready():
    print(f'{client.user} is online now sir')
    try:
        synced = await client.tree.sync()
        print(f'{len(synced)} commands')
    except Exception as e:
        print(f"{e}")

BotToken = "bot token here sir"

class fuckcf:
    def __init__(self):
        self.ua = UserAgent()
        self.session = tls_client.Session(client_identifier="safari_15_6_1")
        self.headers = {
            "Referer": "https://early-access.bloxgame.com",
            "Content-Type": "application/x-www-form-urlencoded",
        }

    def generate_fake_user_agent(self):
        return self.ua.random


class Slide: 
    def __init__(self):
        self.cf_bypass = fuckcf()  
        self.session = self.cf_bypass.session
        self.headers = {
            "x-auth-token": "",
            "Referer": "https://early-access.bloxgame.com",
            "Content-Type": "application/json",
            "User-Agent": self.cf_bypass.generate_fake_user_agent()
        }

    def make_request(self, url):
        headers = self.headers.copy()
        while True:
            try:
                response = self.session.get(url, headers=headers)
                if response.status_code == 200:
                    return response.json()
            except Exception as e:
                print(f"{e}")
                time.sleep(0.01)

    def Algorithm(self):
        game = self.make_request("https://api.bloxgame.com/games/roulette")
        pastwinings = [x['winningColor'] for x in game['history'][:10]][::-1]

        if len(pastwinings) < 10:
            return pastwinings[:4], "none", None

        prediction = pastwinings[9]
        return pastwinings[-3:], prediction, None

    def Algorithm2(self):
        game = self.make_request("https://api.bloxgame.com/games/roulette")
        if not game or 'history' not in game:
            return None

        past = [x['winningColor'] for x in game['history'][:4]][::-1] 
        color_mapping = {'yellow': 3, 'red': 1, 'purple': 2}

        if len(past) < 3:
            return None
        mine = [color_mapping[color] for color in past[2:]]
        X = np.array([[color_mapping[past[i-2]], color_mapping[past[i-1]]] for i in range(2, len(past))])
        X_train, X_test, y_train, y_test = train_test_split(X, mine, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        last_two_winning_colors = [color_mapping[past[-2]], color_mapping[past[-1]]]
        next_color_encoded = model.predict(np.array([last_two_winning_colors]))[0]
        pastwinings = [x['winningColor'] for x in game['history'][:4]][::-1] 
        prediction = [color for color, code in color_mapping.items() if code == next_color_encoded][0]

        return pastwinings[-3:], prediction, None

    def AsuraAlgorithm(self):
        game = self.make_request("https://api.bloxgame.com/games/roulette")
        if not game or 'history' not in game or len(game['history']) < 3:
            return [], "none", None
        past = [x['winningColor'] for x in game['history'][:4]][::-1]
        color_mapping = {'yellow': 3, 'red': 1, 'purple': 2}
        mine = [color_mapping[color] for color in past[2:]]
        X = np.array([[color_mapping[past[i - 2]], color_mapping[past[i - 1]]] for i in range(2, len(past))])
        X_train, X_test, y_train, y_test = train_test_split(X, mine, test_size=0.2, random_state=42)
        model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        last_two_winning_colors = [color_mapping[past[-2]], color_mapping[past[-1]]]
        next_color_encoded = model.predict(np.array([last_two_winning_colors]))[0]
        pastwinings = [x for x in past[:4]]
        prediction = [color for color, code in color_mapping.items() if code == next_color_encoded][0]
        return pastwinings[-3:], prediction, None

    def PatternRecognition(self):
        game = self.make_request("https://api.bloxgame.com/games/roulette")
        if not game or 'history' not in game or len(game['history']) < 4:
            return [], "none", None
        pastwinings = [x['winningColor'] for x in game['history'][:10]][::-1]
        color_mapping = {'yellow': 3, 'red': 1, 'purple': 2}
        reverse_mapping = {v: k for k, v in color_mapping.items()}
        if len(pastwinings) < 4:
            return pastwinings[:4], "none", None
        encoded = [color_mapping[color] for color in pastwinings]
        transitions = {key: {1: 0, 2: 0, 3: 0} for key in [1, 2, 3]}
        for i in range(len(encoded) - 1):
            transitions[encoded[i]][encoded[i + 1]] += 1
        for key, value in transitions.items():
            total = sum(value.values())
            if total > 0:
                transitions[key] = {k: v / total for k, v in value.items()}
        lastgame = encoded[-1]
        probabilities = transitions[lastgame]
        predict = max(probabilities, key=probabilities.get)
        prediction = reverse_mapping[predict]
        return pastwinings[-3:], prediction, None
    
    def smoothpredict(self):
        game = self.make_request("https://api.bloxgame.com/games/roulette")
        if not game or 'history' not in game or len(game['history']) < 4:
            return [], "none", None
        pastwinings = [x['winningColor'] for x in game['history'][:10]][::-1]
        colors = ['red', 'yellow', 'purple']
        r = {color: pastwinings.count(color) for color in colors}
        sorted_colors = sorted(r.items(), key=lambda item: item[1])
        if len(sorted_colors) < 2:
            return pastwinings[-3:], "none", None
        prediction = sorted_colors[1][0] 

        return pastwinings[-3:], prediction, None 

    def get_currentgame(self):
        game = self.make_request("https://api.bloxgame.com/games/roulette")
        current_game = game['current']
        return {
            "uuid": current_game['_id'],
        }


@client.tree.command(name='slide')
@app_commands.choices(
    algorithm=[
        app_commands.Choice(name="Algorithm", value="Algorithm"),
        app_commands.Choice(name="Algorithm2", value="Algorithm2"),
        app_commands.Choice(name="AsuraAlgorithm", value="AsuraAlgorithm"),
        app_commands.Choice(name="PatternRecognition", value="PatternRecognition"),
        app_commands.Choice(name="smoothpredict", value="smoothpredict"),
    ]
)
async def slide(interaction: discord.Interaction, algorithm: str):
    await interaction.response.defer()
    predictor = Slide()
    if algorithm == "Algorithm":
        pastwinings, prediction, _ = predictor.Algorithm()
    elif algorithm == "Algorithm2":
        pastwinings, prediction, _ = predictor.Algorithm2() 
    elif algorithm == "AsuraAlgorithm":
        pastwinings, prediction, _ = predictor.AsuraAlgorithm()
    elif algorithm == "PatternRecognition":
        pastwinings, prediction, _ = predictor.PatternRecognition()
    elif algorithm == "smoothpredict":
        pastwinings, prediction, _ = predictor.smoothpredict()

    game = predictor.get_currentgame()
    uuid = game["uuid"]

    if prediction == "FAILED":
        embed = discord.Embed(
            title="``❌`` something went wrong. please try again",
            color=discord.Color.red()
        )
        await interaction.followup.send(embed=embed)
        return

    color_to_emoji = {"blue": "🔵","purple": "🟣","yellow": "🟡"}
    color_emoji = color_to_emoji.get(prediction, "")
    embed = discord.Embed(title="Asura's Predict `Slide.`",)
    embed.add_field(name="Prediction:", value=f"``{prediction}{color_emoji}``", inline=False)
    embed.add_field(name="Algorithm:", value=f"``{algorithm}``", inline=False)
    embed.add_field(name="LastWinnings", value=f"``{{{', '.join(pastwinings)}}}``", inline=False)
    embed.add_field(name="UUID:", value=f"``{uuid}``", inline=False)
    await interaction.followup.send(embed=embed)



class Crash:
    def __init__(self):
        self.cf_bypass = fuckcf()  
        self.session = self.cf_bypass.session
        self.headers = {
            "x-auth-token": "",
            "Referer": "https://early-access.bloxgame.com",
            "Content-Type": "application/json",
            "User-Agent": self.cf_bypass.generate_fake_user_agent()
        }

    def make_request(self, url):
        headers = self.headers.copy()
        while True:
            try:
                response = self.session.get(url, headers=headers)
                if response.status_code == 200:
                    return response.json()
            except Exception:
                print("Bypass failed, retrying...")
                time.sleep(0.01)

    def get_games(self):
        data = self.make_request("https://api.bloxgame.com/games/crash")
        if data:
            games = [x['crashPoint'] for x in data['history'][:25]][::-1]
            uuid = data['current'].get('_id', "None")
            return games, uuid
        else:
            return [], ""

    def Algortihm(self):
        games, _ = self.get_games()
        if not games:
            return "None", "", ""
        X = []
        y = []
        for i in range(len(games) - 1):
            X.append(games[i])
            y.append(games[i + 1])
        self.model = LinearRegression()
        self.scaler = MinMaxScaler()
        X = np.array(X).reshape(-1, 1)
        y = np.array(y)
        X_scaled = self.scaler.fit_transform(X)
        y_scaled = self.scaler.fit_transform(y.reshape(-1, 1))
        self.model.fit(X_scaled, y_scaled)
        last_game = np.array([games[-1]]).reshape(-1, 1)
        last_game_scaled = self.scaler.transform(last_game)
        prediction_scaled = self.model.predict(last_game_scaled)
        prediction = self.scaler.inverse_transform(prediction_scaled)
        prediction = max(prediction[0][0] * 1, 1) 
        safeprediction = max(prediction * 0.35, 1)
        return prediction , safeprediction

    def HyperNova(self):
        games, _ = self.get_games()
        if not games:
            return "None", "", ""
        X = []
        y = []
        for i in range(len(games) - 1):
            X.append(games[i])
            y.append(games[i + 1])
        X = np.array(X).reshape(-1, 1)
        y = np.array(y)
        self.scaler = MinMaxScaler()
        X_scaled = self.scaler.fit_transform(X)
        y_scaled = self.scaler.fit_transform(y.reshape(-1, 1))
        self.model = GradientBoostingRegressor(n_estimators=100,learning_rate=0.1,max_depth=3,random_state=42)
        self.model.fit(X_scaled, y_scaled.ravel())
        game = np.array([games[-1]]).reshape(-1, 1)
        game_scaled = self.scaler.transform(game)
        prediction_scaled = self.model.predict(game_scaled)
        prediction = self.scaler.inverse_transform(prediction_scaled.reshape(-1, 1))
        prediction = max(prediction[0][0] * 1, 1)
        safeprediction = max(prediction * 0.35, 1)
        return prediction, safeprediction

    def Pulsar(self):
        games, _ = self.get_games()
        if not games:
            return "None", ""
        X = np.array(games[:-1]).reshape(-1, 1)
        y = np.array(games[1:])
        self.scaler = MinMaxScaler()
        X_scaled = self.scaler.fit_transform(X)
        y_scaled = self.scaler.fit_transform(y.reshape(-1, 1))
        self.model = AdaBoostRegressor(n_estimators=100, learning_rate=0.1)
        self.model.fit(X_scaled, y_scaled.ravel())
        game = np.array([games[-1]]).reshape(-1, 1)
        game_scaled = self.scaler.transform(game)
        prediction_scaled = self.model.predict(game_scaled)
        prediction = self.scaler.inverse_transform(prediction_scaled.reshape(-1, 1))
        prediction = max(prediction[0][0] * 1, 1)
        safeprediction = max(prediction * 0.35, 1)
        return prediction, safeprediction

    def VectorMachines(self):
        games, _ = self.get_games()
        if not games:
            return "None", ""
        X = np.array(games[:-1]).reshape(-1, 1)
        y = np.array(games[1:])
        self.scaler = MinMaxScaler()
        X_scaled = self.scaler.fit_transform(X)
        y_scaled = self.scaler.fit_transform(y.reshape(-1, 1))
        self.model = SVR(kernel='rbf', C=100, epsilon=0.1)
        self.model.fit(X_scaled, y_scaled.ravel())
        game = np.array([games[-1]]).reshape(-1, 1)
        game_scaled = self.scaler.transform(game)
        prediction_scaled = self.model.predict(game_scaled)
        prediction = self.scaler.inverse_transform(prediction_scaled.reshape(-1, 1))
        prediction = max(prediction[0][0] * 1, 1)
        safeprediction = max(prediction * 0.35, 1)
        return prediction, safeprediction

    def Hybrid(self):
        predictions = []
        safepredictions = []
        for method in [self.Algortihm, self.HyperNova, self.Pulsar, self.VectorMachines]:
            prediction, safeprediction = method()
            if prediction != "None":
                predictions.append(prediction)
                safepredictions.append(safeprediction)
        if predictions:
            prediction = sum(predictions) / len(predictions)
            safeprediction = sum(safepredictions) / len(safepredictions)
            return prediction, safeprediction
        else:
            return "None", ""


@client.tree.command(name="crash", description="🛝 Predict crash using a selected algorithm.")
@app_commands.choices(
    algorithm=[
        app_commands.Choice(name="Algortihm", value="Algortihm"),
        app_commands.Choice(name="HyperNova", value="HyperNova"),
        app_commands.Choice(name="Pulsar", value="Pulsar"),
        app_commands.Choice(name="VectorMachines", value="VectorMachines"),
        app_commands.Choice(name="Hybrid", value="Hybrid"),
    ]
)
async def crashprediction(interaction: discord.Interaction, algorithm: str):
    await interaction.response.defer()

    predictor = Crash()
    if algorithm == "Algortihm":
        prediction, safeprediction = predictor.Algortihm()
    elif algorithm == "HyperNova":
        prediction, safeprediction = predictor.HyperNova()
    elif algorithm == "Pulsar":
        prediction, safeprediction = predictor.Pulsar()
    elif algorithm == "VectorMachines":
        prediction, safeprediction = predictor.VectorMachines()
    elif algorithm == "Hybrid":
        prediction, safeprediction = predictor.Hybrid()

    games, uuid = predictor.get_games()
    plt.figure(figsize=(10, 6))
    plt.style.use("dark_background")
    plt.plot(range(len(games)), games, marker="o", markersize=10, linewidth=2, label="CrashPoints", color="cyan")
    plt.axhline(y=prediction, color="purple", linestyle="--", linewidth=2, label=f"Prediction: {prediction:.2f}x")
    plt.axhline(y=safeprediction, color="green", linestyle="--", linewidth=2, label=f"Safe Bet: {safeprediction:.2f}x")
    plt.title("Prediction", fontsize=16)
    plt.xlabel("Games Train Data", fontsize=12)
    plt.ylabel("Multiplier", fontsize=12)
    plt.legend(fontsize=14)
    plt.grid(True, color="gray", linestyle="--", alpha=0.5)
    path = "chart.png"
    plt.savefig(path, transparent=True)
    plt.close()

    embed = discord.Embed(title="Asura's Predict `Crash.`",)
    embed.add_field(name="Prediction:", value=f"``{prediction:.2f}x``", inline=False)
    embed.add_field(name="SafeBet:", value=f"``{safeprediction:.2f}x``", inline=False)
    embed.add_field(name="Algorithm", value=f"``{algorithm}``", inline=False)
    embed.add_field(name="UUID", value=f"``{uuid}``", inline=False)
    with open(path, "rb") as file:
        file = discord.File(file, filename="chart.png")
        embed.set_image(url="attachment://chart.png")
        await interaction.followup.send(embed=embed, file=file)
    os.remove(path)


client.run(BotToken)
