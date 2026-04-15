import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
Y = np.array([0, 0, 0, 1], dtype=float)


class Adaline:
	def __init__(self, learning_rate=0.1, epochs=100, seed=42):
		self.learning_rate = learning_rate
		self.epochs = epochs
		rng = np.random.default_rng(seed)
		self.weights = rng.normal(0, 0.5, size=2)
		self.bias = float(rng.normal(0, 0.5))

	def net(self, x):
		return np.dot(x, self.weights) + self.bias

	def predict(self, x):
		return (self.net(x) >= 0.5).astype(int)

	def fit(self, x, y, logger):
		losses = []
		for epoch in range(1, self.epochs + 1):
			outputs = self.net(x)
			errors = y - outputs
			self.weights += self.learning_rate * np.dot(x.T, errors) / len(x)
			self.bias += self.learning_rate * errors.mean()
			mse = float(np.mean(errors ** 2))
			losses.append(mse)
			logger.info("epoch=%03d mse=%.6f weights=%s bias=%.6f", epoch, mse, np.round(self.weights, 5).tolist(), self.bias)
		return losses


def setup_logger(log_dir):
	log_dir.mkdir(parents=True, exist_ok=True)
	log_file = log_dir / f"adaline_and_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
	logger = logging.getLogger("adaline_and")
	logger.handlers.clear()
	logger.setLevel(logging.INFO)
	formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
	file_handler = logging.FileHandler(log_file, encoding="utf-8")
	file_handler.setFormatter(formatter)
	console_handler = logging.StreamHandler()
	console_handler.setFormatter(formatter)
	logger.addHandler(file_handler)
	logger.addHandler(console_handler)
	return logger, log_file


def save_loss_plot(losses, out_file):
	plt.figure(figsize=(7, 4))
	plt.plot(range(1, len(losses) + 1), losses, marker="o")
	plt.title("Adaline AND - Loss")
	plt.xlabel("Epoca")
	plt.ylabel("MSE")
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	plt.savefig(out_file, dpi=150)
	plt.close()


def save_boundary_plot(model, out_file):
	grid_x, grid_y = np.meshgrid(np.linspace(-0.2, 1.2, 200), np.linspace(-0.2, 1.2, 200))
	grid = np.c_[grid_x.ravel(), grid_y.ravel()]
	scores = model.net(grid).reshape(grid_x.shape)
	plt.figure(figsize=(6, 5))
	plt.contourf(grid_x, grid_y, scores, levels=20, cmap="RdYlBu_r", alpha=0.85)
	plt.colorbar(label="Salida lineal")
	plt.scatter(X[:, 0], X[:, 1], c=Y, cmap="coolwarm", s=160, edgecolor="black")
	plt.title("Adaline AND - frontera")
	plt.xlabel("x1")
	plt.ylabel("x2")
	plt.tight_layout()
	plt.savefig(out_file, dpi=150)
	plt.close()


def main():
	base_dir = Path(__file__).resolve().parent
	log_dir = base_dir / "log"
	images_dir = log_dir / "images"
	images_dir.mkdir(parents=True, exist_ok=True)

	logger, log_file = setup_logger(log_dir)
	logger.info("tabla_AND=%s", X.tolist())
	logger.info("salidas=%s", Y.astype(int).tolist())
	logger.info("log_file=%s", log_file)

	model = Adaline(learning_rate=0.1, epochs=100, seed=42)
	logger.info("pesos_iniciales=%s bias_inicial=%.6f", np.round(model.weights, 5).tolist(), model.bias)

	losses = model.fit(X, Y, logger)
	predictions = model.predict(X)
	accuracy = float((predictions == Y).mean())

	logger.info("pesos_finales=%s bias_final=%.6f", np.round(model.weights, 5).tolist(), model.bias)
	logger.info("predicciones=%s", predictions.astype(int).tolist())
	logger.info("accuracy=%.2f%%", accuracy * 100)

	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	loss_file = images_dir / f"loss_{timestamp}.png"
	boundary_file = images_dir / f"boundary_{timestamp}.png"
	save_loss_plot(losses, loss_file)
	save_boundary_plot(model, boundary_file)

	print("Entrenamiento completado.")
	print(f"Log: {log_file}")
	print(f"Imagen pérdida: {loss_file}")
	print(f"Imagen frontera: {boundary_file}")


if __name__ == "__main__":
	main()
