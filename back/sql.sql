create table roles(
    role_id int primary key auto_increment,
    role_name varchar(20) not null
);
insert into roles (role_name) values ('管理员'), ('普通用户');

create table users(
    user_id int primary key auto_increment,
    username varchar(20) not null unique,
    password char(32) not null,
    role_id int not null default 2,
    foreign key(role_id) references roles(role_id)
);
insert into users (username, password, role_id) values ('admin', '202cb962ac59075b964b07152d234b70', 1), ('gerald', '202cb962ac59075b964b07152d234b70', 2);