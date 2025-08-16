-- Enable UUID extension if not already enabled
create extension if not exists "uuid-ossp";

-- Create tables first
create table if not exists time_tracking (
    id uuid default uuid_generate_v4() primary key,
    user_id uuid references auth.users(id) on delete cascade,
    session_id text not null,
    time_spent integer not null,
    session_start timestamp with time zone,
    session_end timestamp with time zone,
    end_reason text,
    created_at timestamp with time zone default now()
);

create table if not exists user_goals (
    id uuid default uuid_generate_v4() primary key,
    user_id uuid references auth.users(id) on delete cascade unique,
    daily_goal_seconds integer not null default 3600,
    created_at timestamp with time zone default now(),
    updated_at timestamp with time zone default now()
);

-- Enable Row Level Security
alter table time_tracking enable row level security;
alter table user_goals enable row level security;

-- Create policies after tables exist
create policy "Users can view their own time tracking"
    on time_tracking for select
    using (auth.uid() = user_id);

create policy "Users can insert their own time tracking"
    on time_tracking for insert
    with check (auth.uid() = user_id);

create policy "Users can view their own goals"
    on user_goals for select
    using (auth.uid() = user_id);

create policy "Users can insert their own goals"
    on user_goals for insert
    with check (auth.uid() = user_id);

create policy "Users can update their own goals"
    on user_goals for update
    using (auth.uid() = user_id);

-- Create indexes after tables and policies
create index if not exists time_tracking_user_id_idx on time_tracking(user_id);
create index if not exists time_tracking_created_at_idx on time_tracking(created_at);
create index if not exists user_goals_user_id_idx on user_goals(user_id);

-- Enable realtime for time_tracking table
begin;
  drop publication if exists supabase_realtime;
  create publication supabase_realtime;
commit;

alter publication supabase_realtime add table time_tracking;