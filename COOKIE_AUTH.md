# Cookie-Based Authentication System

This FastAPI application now includes cookie-based authentication for secure user sessions.

## How it Works

1. **Login**: User sends Google ID token → Server validates and returns JWT cookie
2. **Session**: Browser automatically sends cookie with each request
3. **Logout**: Server clears the cookie from browser
4. **Protection**: Protected endpoints require valid cookie

## API Endpoints

### Authentication Endpoints

#### POST `/login`
- **Purpose**: Login with Google ID token and receive authentication cookie
- **Headers**: `Authorization: Bearer <google_id_token>`
- **Response**: Sets HTTP-only cookie and returns user info
- **Cookie**: `auth_token` (HTTP-only, Secure, SameSite=strict)

#### POST `/logout`
- **Purpose**: Logout and clear authentication cookie
- **Response**: Clears the `auth_token` cookie

#### GET `/me`
- **Purpose**: Get current user information
- **Authentication**: Requires valid cookie
- **Response**: User email, name, and picture

### Protected Endpoints

#### POST `/analyze_data`
- **Purpose**: Analyze data with file uploads
- **Authentication**: Requires valid cookie (replaces Bearer token)
- **Files**: `questions.txt` (required) + other attachments

## Security Features

### Cookie Security
```python
response.set_cookie(
    key="auth_token",
    value=jwt_token,
    max_age=7*24*3600,     # 7 days
    httponly=True,         # Prevents XSS attacks
    secure=True,           # HTTPS only
    samesite="strict"      # CSRF protection
)
```

### JWT Token Structure
```json
{
  "email": "user@example.com",
  "name": "User Name",
  "picture": "https://...",
  "exp": 1640995200,
  "iat": 1640908800
}
```

## Frontend Integration

### JavaScript Example
```javascript
// Login (after getting Google token)
const response = await fetch('/login', {
    method: 'POST',
    headers: {
        'Authorization': `Bearer ${googleToken}`
    },
    credentials: 'include'  // Important: includes cookies
});

// Make authenticated requests
const userInfo = await fetch('/me', {
    credentials: 'include'  // Automatically sends cookie
});

// Logout
await fetch('/logout', {
    method: 'POST',
    credentials: 'include'
});
```

### HTML Form Example
```html
<!-- File upload form (cookies sent automatically) -->
<form action="/analyze_data" method="post" enctype="multipart/form-data">
    <input type="file" name="questions.txt" required>
    <input type="file" name="data.csv" multiple>
    <button type="submit">Analyze</button>
</form>
```

## Environment Variables

```env
GOOGLE_CLIENT_ID=your_google_client_id
JWT_SECRET_KEY=your_secret_key_for_jwt  # Optional, auto-generated if not set
```

## Cookie Storage

- **Location**: Stored in user's browser
- **Lifetime**: 7 days (configurable)
- **Security**: HTTP-only, Secure, SameSite protection
- **Automatic**: Browser handles sending/receiving

## Migration from Bearer Tokens

### Before (Bearer Token)
```python
# Client had to manage token
headers = {"Authorization": f"Bearer {token}"}
response = requests.post("/analyze_data", headers=headers, files=files)
```

### After (Cookies)
```python
# Browser automatically handles authentication
session = requests.Session()  # Handles cookies automatically
session.post("/login", headers={"Authorization": f"Bearer {google_token}"})
response = session.post("/analyze_data", files=files)  # Cookie sent automatically
```

## Benefits

1. **Automatic**: No manual token management needed
2. **Secure**: HTTP-only cookies prevent XSS attacks
3. **Convenient**: Works with forms and AJAX
4. **Standard**: Web-native authentication method
5. **Persistent**: Survives browser refresh/navigation

## Testing

Run the example script:
```bash
python auth_example.py
```

The script demonstrates the complete authentication flow with cookie handling.
