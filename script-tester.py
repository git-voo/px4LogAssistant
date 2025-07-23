from google import genai
# import google.generativeai as genai

model = genai.GenerativeModel("models/gemini-pro:generateContent")

response =model.generate_content(
    model="gemini-2.5-flash",
    contents="Explain how AI works in a few words",
)

print(response.text)
