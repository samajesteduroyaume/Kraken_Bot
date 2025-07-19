import typer
import yaml
from pathlib import Path
import requests

CONFIG_PATH = Path('config/config.yaml')
app = typer.Typer()

def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

def save_config(config):
    with open(CONFIG_PATH, 'w') as f:
        yaml.dump(config, f, sort_keys=False, allow_unicode=True)

@app.command('list')
def list_pairs():
    "Affiche les paires configurées."
    config = load_config()
    pairs = config['trading']['pairs']
    typer.echo("Paires configurées :")
    for p in pairs:
        typer.echo(f"- {p}")

@app.command()
def add_pair(pair: str):
    "Ajoute une paire à la config."
    config = load_config()
    if pair not in config['trading']['pairs']:
        config['trading']['pairs'].append(pair)
        save_config(config)
        typer.echo(f"✅ Paire ajoutée : {pair}")
    else:
        typer.echo(f"⚠️  Paire déjà présente : {pair}")

@app.command()
def remove_pair(pair: str):
    "Retire une paire de la config."
    config = load_config()
    if pair in config['trading']['pairs']:
        config['trading']['pairs'].remove(pair)
        save_config(config)
        typer.echo(f"❌ Paire retirée : {pair}")
    else:
        typer.echo(f"⚠️  Paire non trouvée : {pair}")

@app.command()
def set_criteria(min_score: float = typer.Option(None, help="Score minimum"),
                 min_volume: int = typer.Option(None, help="Volume minimum")):
    "Modifie les critères de sélection."
    config = load_config()
    if min_score is not None:
        config['trading']['min_score'] = min_score
        typer.echo(f"Score minimum mis à jour : {min_score}")
    if min_volume is not None:
        config['trading']['min_volume'] = min_volume
        typer.echo(f"Volume minimum mis à jour : {min_volume}")
    save_config(config)

@app.command()
def show():
    "Affiche la config complète."
    config = load_config()
    typer.echo(yaml.dump(config, allow_unicode=True, sort_keys=False))

@app.command()
def add_all_pairs(api_url: str = typer.Option('https://api.kraken.com/0/public/AssetPairs', help="URL de l'API Kraken")):
    "Ajoute toutes les paires Kraken disponibles à la config."
    typer.echo("Récupération des paires Kraken...")
    resp = requests.get(api_url)
    data = resp.json()
    if 'result' not in data:
        typer.echo("Erreur lors de la récupération des paires Kraken.")
        raise typer.Exit(1)
    all_pairs = list(data['result'].keys())
    config = load_config()
    config['trading']['pairs'] = all_pairs
    save_config(config)
    typer.echo(f"✅ {len(all_pairs)} paires ajoutées à la config.")

if __name__ == "__main__":
    app() 