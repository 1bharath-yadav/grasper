"""
Example of how to use the cookie-based authentication system
"""
import requests
import json

# Base URL of your FastAPI app
BASE_URL = "http://localhost:8000"


def example_auth_flow():
    """
    Example authentication flow using cookies
    """

    # Step 1: Login with Google token
    # Note: In a real app, you'd get this token from Google OAuth flow
    google_token = "your_google_id_token_here"

    # Create a session to automatically handle cookies
    session = requests.Session()

    try:
        # Step 2: Login and get cookie
        login_response = session.post(
            f"{BASE_URL}/login",
            headers={"Authorization": f"Bearer {google_token}"}
        )

        if login_response.status_code == 200:
            print("✅ Login successful!")
            print(f"Response: {login_response.json()}")

            # The cookie is now automatically stored in the session

            # Step 3: Make authenticated requests (cookies sent automatically)
            me_response = session.get(f"{BASE_URL}/me")
            if me_response.status_code == 200:
                print("✅ Got user info from cookie:")
                print(f"User: {me_response.json()}")

            # Step 4: Make an analysis request (protected endpoint)
            # Note: This would need actual file data in practice
            files = {
                "questions.txt": ("questions.txt", "What is the main trend in the data?", "text/plain")
            }

            analysis_response = session.post(
                f"{BASE_URL}/analyze_data",
                files=files
            )

            if analysis_response.status_code == 200:
                print("✅ Analysis request successful!")
            else:
                print(f"❌ Analysis failed: {analysis_response.status_code}")
                print(analysis_response.text)

            # Step 5: Logout (clears cookie)
            logout_response = session.post(f"{BASE_URL}/logout")
            if logout_response.status_code == 200:
                print("✅ Logout successful!")

            # Step 6: Try to access protected endpoint after logout (should fail)
            me_response_after_logout = session.get(f"{BASE_URL}/me")
            if me_response_after_logout.status_code == 401:
                print("✅ Protected endpoint correctly denied access after logout")

        else:
            print(f"❌ Login failed: {login_response.status_code}")
            print(login_response.text)

    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")


if __name__ == "__main__":
    print("🔐 Cookie-based Authentication Example")
    print("=====================================")
    example_auth_flow()
