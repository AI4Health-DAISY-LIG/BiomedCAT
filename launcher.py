import tkinter as tk
from tkinter import scrolledtext, messagebox
import subprocess
import threading
import os
import platform
import queue
import time

class BiomedCATLauncher(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("BiomedCAT Launcher")
        self.geometry("700x500")
        self.configure(bg="#2c3e50")

        # Configuration des chemins et commandes
        self.model_name = "llama3.1"
        self.docker_compose_cmd = ["docker", "compose", "up", "-d"]
        self.docker_compose_down_cmd = ["docker", "compose", "down"]
        self.log_cmd = ["docker", "logs", "-f", "biomedcat_pipeline"]
        self.data_dir = os.path.abspath("data")

        # File d'attente pour la communication entre le thread de log et l'interface
        self.log_queue = queue.Queue()

        self._setup_ui()
        
        # Lancement de la vérification initiale au démarrage
        self.after(100, self.check_environment)
        
        # Lancement du thread qui surveille la file d'attente des logs pour mettre à jour l'UI
        self.after(100, self._poll_logs)

    def _setup_ui(self):
        """Initialise l'interface graphique."""
        # Header
        header = tk.Label(self, text="BiomedCAT Control Center", font=("Helvetica", 18, "bold"), 
                          bg="#2c3e50", fg="#ecf0f1", pady=20)
        header.pack()

        # Zone de texte pour les logs
        self.log_area = scrolledtext.ScrolledText(self, wrap=tk.WORD, bg="#1e272e", fg="#d2dae2", 
                                                 font=("Consolas", 10), state='disabled')
        self.log_app_frame = tk.Frame(self, bg="#2c3e50")
        self.log_app_frame.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)
        self.log_area.pack(fill=tk.BOTH, expand=True)

        # Frame pour les boutons de contrôle
        btn_frame = tk.Frame(self, bg="#2c3e50", pady=20)
        btn_frame.pack(fill=tk.X)

        self.btn_start = tk.Button(btn_frame, text="🚀 Démarrer", command=self.start_services, 
                                   bg="#27ae60", fg="white", font=("Helvetica", 10, "bold"), width=15)
        self.btn_start.pack(side=tk.LEFT, padx=10, expand=True)

        self.btn_stop = tk.Button(btn_frame, text="🛑 Arrêter", command=self.stop_services, 
                                  bg="#c0392b", fg="white", font=("Helvetica", 10, "bold"), width=15)
        self.btn_stop.pack(side=tk.LEFT, padx=10, expand=True)

        self.btn_data = tk.Button(btn_frame, text="📂 Dossier Data", command=self.open_data_folder, 
                                  bg="#2980b9", fg="white", font=("Helvetica", 10, "bold"), width=15)
        self.btn_data.pack(side=tk.LEFT, padx=10, expand=True)

        # Barre de statut
        self.status_var = tk.StringVar(value="Prêt")
        status_bar = tk.Label(self, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, 
                             anchor=tk.W, bg="#34495e", fg="#ecf0f1")
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _write_log(self, message):
        """Ajoute un message dans la zone de texte des logs."""
        self.log_area.configure(state='normal')
        self.log_area.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {message}\n")
        self.log_area.see(tk.END)
        self.log_area.configure(state='disabled')

    def _poll_logs(self):
        """Vérifie la file d'attente pour mettre à jour l'interface avec les nouveaux logs."""
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self._write_log(msg)
        except queue.Empty:
            pass
        finally:
            self.after(100, self._poll_logs)

    def check_environment(self):
        """Vérifie si Docker et Ollama sont installés."""
        self._write_log("Vérification de l'environnement...")
        
        docker_ok = self._check_cmd("docker --version")
        ollama_ok = self._check_cmd("ollama --version")

        if not docker_ok:
            messagebox.showerror("Erreur", "Docker n'est pas installé.\nTéléchargez-le ici : https://www.docker.com/products/docker-desktop")
            self.status_var.set("Erreur: Docker manquant")
        elif not ollama_ok:
            messagebox.showerror("Erreur", "Ollama n'est pas installé.\nTéléchargez-le ici : https://ollama.com/")
            self.status_var.set("Erreur: Ollama manquant")
        else:
            self._write_log("✅ Docker et Ollama sont prêts.")
            self.status_var.set("Environnement prêt")
            # Vérifier si les modèles sont présents
            threading.Thread(target=self.check_ollama_model, daemon=True).start()

    def _check_cmd(self, cmd):
        """Vérifie si une commande système est disponible."""
        try:
            subprocess.run(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def check_ollama_model(self):
        """Vérifie si les modèles Ollama sont téléchargés, sinon lance les pulls."""
        self._write_log("Vérification des modèles Ollama...")
        
        # Liste de tous les modèles nécessaires
        required_models = [
            self.model_name,  # "llama3.1"
            "gemma4:e4b-it-qat",  # classification_model_id
            "llama-guard3:8b",    # sanitization_model_id
            "qwen2.5vl:7b"        # ocr_model_id
        ]
        
        try:
            # Vérifier qu'Ollama est disponible
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            
            missing_models = []
            for model in required_models:
                if model not in result.stdout:
                    missing_models.append(model)
            
            if missing_models:
                self._write_log(f"📥 Modèles manquants : {missing_models}. Téléchargement en cours...")
                self.status_var.set("Téléchargement des modèles...")
                
                # Télécharger chaque modèle manquant
                for model in missing_models:
                    self._write_log(f" Téléchargement du modèle : {model}")
                    self._run_process_to_queue(["ollama", "pull", model])
            else:
                self._write_log("✅ Tous les modèles Ollama sont prêts.")
                self.status_var.set("Modèles prêts")
                
            # Télécharger le modèle RAG si nécessaire
            self._download_rag_model()
            
        except Exception as e:
            self._write_log(f"❌ Erreur Ollama: {str(e)}")

    def _download_rag_model(self):
        """Télécharge le modèle RAG locallement."""
        try:
            # Importer ici pour éviter les problèmes d'importation circulaire
            from biomedcat.config import Settings
            from biomedcat.stages.rag_engine import download_model_if_needed
            import os
            
            settings = Settings()
            model_name = settings.RAG_embedding_model
            model_cache_dir = os.path.join(settings.internal_data_path, "models")
            os.makedirs(model_cache_dir, exist_ok=True)
            
            self._write_log(f"Vérification du modèle RAG : {model_name}")
            local_model_path = download_model_if_needed(model_name, model_cache_dir)
            
            if local_model_path:
                self._write_log(f"✅ Modèle RAG téléchargé localement : {local_model_path}")
            else:
                self._write_log(f"⚠️  Impossible de télécharger le modèle RAG : {model_name}")
                
        except Exception as e:
            self._write_log(f"❌ Erreur téléchargement modèle RAG : {str(e)}")

    def start_services(self):
        """Lance les conteneurs Docker."""
        self._write_log("🚀 Démarrage des services Docker...")
        self.status_var.set("Démarrage en cours...")
        threading.Thread(target=self._run_docker_up, daemon=True).start()

    def _run_docker_up(self):
        try:
            # On lance docker compose up -d
            process = subprocess.Popen(self.docker_compose_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            self._run_process_to_queue_from_popen(process)
            self._write_log("✅ Services lancés avec succès.")
            self.status_var.set("Services en cours d'exécution")
        except Exception as e:
            self._write_log(f"❌ Erreur lors du démarrage: {str(e)}")
            self.status_var.set("Erreur de démarrage")

    def stop_services(self):
        """Arrête les conteneurs Docker."""
        self._write_log("🛑 Arrêt des services...")
        try:
            subprocess.run(self.docker_compose_down_cmd, check=True)
            self._write_log("✅ Services arrêtés.")
            self.status_var.set("Services arrêtés")
        except Exception as e:
            self._write_log(f"❌ Erreur lors de l'arrêt: {str(e)}")

    def open_data_folder(self):
        """Ouvre le dossier data dans l'explorateur de fichiers."""
        path = os.path.abspath("data")
        if not os.path.exists(path):
            os.makedirs(path)
        
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(["open", path])
        else:  # Linux
            subprocess.run(["xdg-open", path])
        self._write_log(f"📂 Dossier ouvert : {path}")

    def _run_process_to_queue(self, cmd):
        """Exécute une commande et envoie la sortie vers la queue de logs."""
        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            self._run_process_to_queue_from_popen(process)
        except Exception as e:
            self._write_log(f"❌ Erreur commande: {str(e)}")

    def _run_process_to_queue_from_popen(self, process):
        """Lit la sortie d'un processus Popen et l'envoie dans la queue."""
        def stream_reader():
            for line in iter(process.stdout.readline, ''):
                if line:
                    self.log_queue.put(line.strip())
            process.stdout.close()

        threading.Thread(target=stream_reader, daemon=True).start()

    def _run_process_to_queue(self, cmd):
        """Version simplifiée pour les commandes bloquantes."""
        try:
            output = subprocess.check_output(cmd, text=True)
            for line in output.splitlines():
                self.log_queue.put(line)
        except Exception as e:
            self.log_queue.put(f"Erreur: {str(e)}")

    def _start_log_monitoring(self):
        """Lance le monitoring des logs Docker en arrière-plan."""
        def monitor():
            try:
                # On utilise docker logs -f pour suivre le flux en temps réel
                process = subprocess.Popen(self.log_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                self._run_process_to_queue_from_popen(process)
            except Exception as e:
                self.log_queue.put(f"❌ Erreur monitoring logs: {str(e)}")

        threading.Thread(target=monitor, daemon=True).start()

    # Override start_services to also start log monitoring
    def start_services(self):
        self._write_log("🚀 Démarrage des services Docker...")
        self.status_var.set("Démarrage en cours...")
        self._start_log_monitoring() # On lance le suivi des logs dès qu'on démarre
        threading.Thread(target=self._run_docker_up, daemon=True).start()

if __name__ == "__main__":
    app = BiomedCATLauncher()
    app.mainloop()
