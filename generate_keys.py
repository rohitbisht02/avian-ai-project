# generate_keys.py (Final Corrected Version)

# This import path is the correct one for your version of the library
from streamlit_authenticator.utilities.hasher import Hasher

# The plain-text passwords you want to hash
passwords_to_hash = ['abc', 'def']

# Generate the hashed versions
hashed_passwords = Hasher(passwords_to_hash).generate()

# Print the final list to the console
print("✅ Success! Copy the following list and paste it into your config.yaml file:")
print(hashed_passwords)