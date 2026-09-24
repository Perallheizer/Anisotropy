from scipy.stats import norm

while True:
    p = input("\nEnter p-value (or 'q' to quit): ").strip()

    if p.lower() in ["q", "quit", "exit"]:
        break

    try:
        p = float(p)

        if not (0 < p < 1):
            print("Please enter a p-value between 0 and 1.")
            continue

        sigma = norm.isf(p)  # one-sided conversion

        print(f"p = {p:.6g}  -->  {sigma:.4f} sigma")

    except ValueError:
        print("Invalid input. Please enter a number or 'q' to quit.")
