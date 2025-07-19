import subprocess
import sys
import os

def run_command(cmd, check=True):
    print(f"Exécution : {cmd}")
    result = subprocess.run(cmd, shell=True)
    if check and result.returncode != 0:
        print(f"Erreur lors de l'exécution : {cmd}")
        sys.exit(result.returncode)

def main():
    # Exemple d'étapes de setup (à adapter selon automate_setup.sh)
    print("Démarrage du setup automatique Python...")
    # Vérification de la configuration
    if os.path.exists('scripts/check_config.py'):
        run_command('python scripts/check_config.py')
    # Initialisation de la base de données
    if os.path.exists('scripts/init_db.py'):
        run_command('python scripts/init_db.py')
    # Nettoyage du cache Redis
    if os.path.exists('scripts/cleanup_redis_cache.py'):
        run_command('python scripts/cleanup_redis_cache.py')
    print("Setup automatique terminé.")

if __name__ == "__main__":
    main() 