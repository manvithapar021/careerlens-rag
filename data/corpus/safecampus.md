# SafeCampus — Real-Time Student Safety Platform

SafeCampus is a full-stack student safety platform built with Python,
Flask, SQLAlchemy, Leaflet.js and Bootstrap 5, deployed on Railway.
It has separate student and parent authentication, dashboards, live
tracking, SOS alerts and emergency contacts. The Flask backend serves
data through HTTP routes as REST APIs.

User accounts and sessions use Flask-Login with secure sessions and
password-reset workflows. Automated email alerts are sent with
Flask-Mail. Data is modelled with SQLAlchemy over a relational
database, with CRUD operations for users, contacts and alerts.

Geofencing is implemented with the Haversine formula to trigger alerts
when a student moves beyond campus boundaries. Leaflet.js provides the
interactive map, and Bootstrap 5 provides the responsive interface.
