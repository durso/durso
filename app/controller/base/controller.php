<?php

namespace app\controller\base; 
use app\view\view;
use library\utils;


class controller{
    protected $query;
    protected $action;
    protected $controller;

    
    
    protected function __construct($action,$query) {
        $this->controller = $this->getControllerName();
        $this->query = $query;
        $this->action = $action;
        $this->$action();
    }
    
    
    protected function getControllerName(){
        return basename(str_replace('\\', '/', get_class($this)));
    }
    
    public function __toString(){
        return $this->getControllerName();
    }

}