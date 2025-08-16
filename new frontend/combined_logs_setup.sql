-- Enable UUID extension if not already enabled
create extension if not exists "uuid-ossp";

-- Create agent_logs table for tracking agent sessions
create table if not exists agent_logs (
    id uuid default uuid_generate_v4() primary key,
    user_id text not null, -- Using text to support both UUID and 'guest-user'
    agent_id integer not null,
    agent_name text,
    agent_type text,
    action_type text not null, -- 'start', 'stop', 'reset'
    start_time timestamp with time zone default now(),
    end_time timestamp with time zone,
    duration integer, -- Duration in seconds
    status text default 'active', -- 'active', 'completed', 'interrupted'
    created_at timestamp with time zone default now()
);

-- Create chat_logs table for tracking chatbot and summarizer activities
create table if not exists chat_logs (
    id uuid default uuid_generate_v4() primary key,
    user_id text not null, -- Using text to support both UUID and 'guest-user'
    message text, -- User's message or file name
    model text, -- AI model used (e.g., 'grok', 'llama', 'chatgpt')
    response_length integer, -- Length of AI response
    file_type text, -- Type of file for document summaries
    file_size integer, -- Size of file in bytes
    has_audio boolean, -- Whether audio was generated for document summaries
    action_type text not null, -- 'chat_message', 'document_summary'
    start_time timestamp with time zone default now(),
    status text default 'completed', -- 'completed', 'failed'
    created_at timestamp with time zone default now()
);

-- Enable Row Level Security for both tables
alter table agent_logs enable row level security;
alter table chat_logs enable row level security;

-- Create policies for agent_logs
create policy "Users can view their own agent logs"
    on agent_logs for select
    using (auth.uid()::text = user_id or user_id = 'guest-user');

create policy "Users can insert their own agent logs"
    on agent_logs for insert
    with check (auth.uid()::text = user_id or user_id = 'guest-user');

create policy "Users can update their own agent logs"
    on agent_logs for update
    using (auth.uid()::text = user_id or user_id = 'guest-user');

-- Create policies for chat_logs
create policy "Users can view their own chat logs"
    on chat_logs for select
    using (auth.uid()::text = user_id or user_id = 'guest-user');

create policy "Users can insert their own chat logs"
    on chat_logs for insert
    with check (auth.uid()::text = user_id or user_id = 'guest-user');

create policy "Users can update their own chat logs"
    on chat_logs for update
    using (auth.uid()::text = user_id or user_id = 'guest-user');

-- Create indexes for agent_logs
create index if not exists agent_logs_user_id_idx on agent_logs(user_id);
create index if not exists agent_logs_agent_id_idx on agent_logs(agent_id);
create index if not exists agent_logs_created_at_idx on agent_logs(created_at);
create index if not exists agent_logs_status_idx on agent_logs(status);

-- Create indexes for chat_logs
create index if not exists chat_logs_user_id_idx on chat_logs(user_id);
create index if not exists chat_logs_action_type_idx on chat_logs(action_type);
create index if not exists chat_logs_created_at_idx on chat_logs(created_at);
create index if not exists chat_logs_model_idx on chat_logs(model);

-- Check if realtime publication exists, if not create it
do $$
begin
  if not exists (select 1 from pg_publication where pubname = 'supabase_realtime') then
    create publication supabase_realtime;
  end if;
end
$$;

-- Add both tables to realtime publication
alter publication supabase_realtime add table agent_logs;
alter publication supabase_realtime add table chat_logs;
