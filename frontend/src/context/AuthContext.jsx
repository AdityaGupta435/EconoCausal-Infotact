import { createContext, useContext, useState, useEffect } from "react";

const AuthContext = createContext(null);

const STORAGE_KEY_TOKEN = "econocausal_token";
const STORAGE_KEY_USER = "econocausal_user";

export function AuthProvider({ children }) {
  const [token, setToken] = useState(() => localStorage.getItem(STORAGE_KEY_TOKEN) || null);
  const [user, setUser] = useState(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY_USER);
      return saved ? JSON.parse(saved) : null;
    } catch {
      return null;
    }
  });
  const [isLoading, setIsLoading] = useState(true);

  // Validate or re-hydrate user profile on mount
  useEffect(() => {
    async function verifyAuth() {
      const savedToken = localStorage.getItem(STORAGE_KEY_TOKEN);
      if (!savedToken) {
        setIsLoading(false);
        return;
      }

      try {
        const res = await fetch("/api/auth/me", {
          headers: {
            Authorization: `Bearer ${savedToken}`,
          },
        });

        if (res.ok) {
          const userData = await res.json();
          setUser(userData);
          localStorage.setItem(STORAGE_KEY_USER, JSON.stringify(userData));
        } else {
          // Token expired or invalid
          logout();
        }
      } catch (err) {
        console.warn("Could not verify auth with backend:", err);
      } finally {
        setIsLoading(false);
      }
    }

    verifyAuth();
  }, []);

  const login = async (usernameOrEmail, password) => {
    const res = await fetch("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        username_or_email: usernameOrEmail,
        password: password,
      }),
    });

    if (!res.ok) {
      const errorData = await res.json().catch(() => ({}));
      throw new Error(errorData.detail || "Authentication failed. Check your credentials.");
    }

    const data = await res.json();
    setToken(data.access_token);
    setUser(data.user);
    localStorage.setItem(STORAGE_KEY_TOKEN, data.access_token);
    localStorage.setItem(STORAGE_KEY_USER, JSON.stringify(data.user));
    return data.user;
  };

  const register = async ({ email, username, password, fullName, role }) => {
    const res = await fetch("/api/auth/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        email,
        username,
        password,
        full_name: fullName || null,
        role: role || "analyst",
      }),
    });

    if (!res.ok) {
      const errorData = await res.json().catch(() => ({}));
      throw new Error(errorData.detail || "Registration failed.");
    }

    // Auto login after registration
    return await login(username, password);
  };

  const logout = () => {
    setToken(null);
    setUser(null);
    localStorage.removeItem(STORAGE_KEY_TOKEN);
    localStorage.removeItem(STORAGE_KEY_USER);
    fetch("/api/auth/logout", { method: "POST" }).catch(() => {});
  };

  const hasRole = (allowedRoles) => {
    if (!user || !user.role) return false;
    const list = Array.isArray(allowedRoles) ? allowedRoles : [allowedRoles];
    return list.map((r) => r.toLowerCase()).includes(user.role.toLowerCase());
  };

  const value = {
    user,
    token,
    role: user?.role || null,
    isAuthenticated: !!token && !!user,
    isLoading,
    login,
    register,
    logout,
    hasRole,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
