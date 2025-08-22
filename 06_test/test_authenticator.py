import pytest
import authenticator

def test_authenticator():
    auth = authenticator.Authenticator()

    auth.register("user1", "password1")
    assert "user1" in auth.users

    with pytest.raises(ValueError, match="エラー: ユーザーは既に存在します。"):
        auth.register("user1", "password1")

    assert auth.login("user1", "password1") == "ログイン成功"

    with pytest.raises(ValueError, match="エラー: ユーザー名またはパスワードが正しくありません。"):
        auth.login("user1", "wrongpassword")
