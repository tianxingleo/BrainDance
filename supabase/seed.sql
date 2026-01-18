--
-- PostgreSQL database dump
--

\restrict kGxqZE8wgeMnigaW6QhRzyGsdesUNUfORmWokvKHdPEjVFyaNDF27FIUwTE9Idz

-- Dumped from database version 17.6
-- Dumped by pg_dump version 17.6

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Data for Name: buckets; Type: TABLE DATA; Schema: storage; Owner: supabase_storage_admin
--

COPY storage.buckets (id, name, owner, created_at, updated_at, public, avif_autodetection, file_size_limit, allowed_mime_types, owner_id, type) FROM stdin;
braindance-assets	braindance-assets	\N	2026-01-12 02:44:23.998992+00	2026-01-12 02:44:23.998992+00	f	f	\N	\N	\N	STANDARD
\.


--
-- PostgreSQL database dump complete
--

\unrestrict kGxqZE8wgeMnigaW6QhRzyGsdesUNUfORmWokvKHdPEjVFyaNDF27FIUwTE9Idz

