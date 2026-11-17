# Secure File Vault

A Flask and SQLAlchemy file-sharing application with AES encryption.
Files are encrypted with Fernet before being written to disk, so
stored files are unreadable without the vault key. Logins use
Werkzeug password hashing.

Sharing links expire after a chosen period from one hour to thirty
days, carry download limits, and can be protected with a hashed share
password. An audit log records uploads, downloads and share events.
Uploads are limited to 16 MB and executable files are rejected.
