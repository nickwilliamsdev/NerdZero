with open("scripts/train_arc.py", "r") as f:
    text = f.read()

text = text.replace(
    'print(winner)',
    'print(winner)\n    \n    # Save the winner safely to be loaded by evaluate_agent.py\n    import pickle\n    winner_path = os.path.abspath(os.path.join(local_dir, "../best_genome.pkl"))\n    with open(winner_path, "wb") as f:\n        pickle.dump(winner, f)\n    print(f"Saved best genome to {winner_path}")'
)

with open("scripts/train_arc.py", "w") as f:
    f.write(text)
