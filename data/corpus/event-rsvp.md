# Event RSVP and QR Check-in

A Flask and SQLAlchemy web application for event registrations with
QR-code tickets and one-time check-in. Guests register through the
web interface, receive a QR ticket, and are checked in once at the
door — a second scan of the same ticket is refused.

The backend uses Flask with SQLAlchemy over a relational database,
with CRUD operations for events and guests. A Pandas dashboard shows
registrations and attendance, and attendance can be exported to
Excel. QR codes are generated in Python.
