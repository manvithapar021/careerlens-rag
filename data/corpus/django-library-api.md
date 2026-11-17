# Django Library API

A library management REST API built with Python, Django 5 and Django
REST Framework, with token authentication. It models books, members
and loans. Borrowing refuses when no copies are available, and
returns compute an overdue fine of 10 rupees per day.

The API supports filtering books by author and availability, listing
overdue loans, and a stats endpoint for totals and fines. All models
are registered in the Django admin. The project has 15 automated
tests, all passing, covering borrowing rules, returns, fines and
permissions. The database is SQLite and the data layer uses Django
models with CRUD operations through the REST API.
