<?php
namespace app;
use app\router;


class bootstrap{

    public function start(){

        router::addRoute("/medico/id:int", array("controller" => "medico","action" =>"index"));

    }
}

 
