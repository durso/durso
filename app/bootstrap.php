<?php
namespace app;
use app\router;


class bootstrap{

    public function start(){

        router::addRoute("/amor/:int/:alnum", array("controller" => "index","action" =>"index"));

    }
}

 
