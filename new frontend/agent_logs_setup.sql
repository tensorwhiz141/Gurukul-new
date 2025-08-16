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

-- Enable Row Level Security
alter table agent_logs enable row level security;

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

-- Create indexes for better performance
create index if not exists agent_logs_user_id_idx on agent_logs(user_id);
create index if not exists agent_logs_agent_id_idx on agent_logs(agent_id);
create index if not exists agent_logs_created_at_idx on agent_logs(created_at);
create index if not exists agent_logs_status_idx on agent_logs(status);

-- Add agent_logs to realtime publication
alter publication supabase_realtime add table agent_logs;
