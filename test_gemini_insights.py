
from gemini_insights import generate_customer_insights

sample_reviews = [
    "The package arrived damaged.",
    "Delivery took too long.",
    "The packaging was poor.",
    "The product quality was excellent."
]

result = generate_customer_insights(sample_reviews)
print(result)
