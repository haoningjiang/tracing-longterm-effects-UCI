# from openai import OpenAI

# client = 

# response = client.chat.completions.create(
#     model="gemini-2.0-flash",
#     n=1,
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {
#             "role": "user",
#             "content": "Explain to me how AI works"
#         }
#     ]
# )

# print(response.choices[0].message)


import instructor
from pydantic import BaseModel
from openai import OpenAI

# Define your desired output structure
class ExtractUser(BaseModel):
    name: str
    age: int

# Patch the OpenAI client
client = instructor.from_openai(OpenAI(
    api_key="AIzaSyD-fmZoBrI-XKXh4G-wefsyGdvyWvlB6DQ",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
))

# Extract structured data from natural language
res = client.chat.completions.create(
    model="gemini-2.0-flash",
    response_model=ExtractUser,
    messages=[{"role": "user", "content": "John Doe is 30 years old."}],
)

print(res.name)  # Output: John Doe
print(res.age)   # Output: 30