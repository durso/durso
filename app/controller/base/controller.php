<?php

namespace app\controller\base; 
use app\view\view;
use app\request;
use app\model\file;
use library\utils;
use library\layout\layout;
use library\layout\elements\element;
use library\layout\elements\script;



class controller{
    protected $query;
    protected $action;
    protected $render = array();
    protected $errorMsg = array();
    protected $error;
    protected $layout;
    protected $view;
    protected $controller;
    
    protected function __construct($action,$query) {
        $this->controller = $this->getControllerName();
        $this->query = $query;
        $this->action = $action;
        $this->layout = new layout();
        $this->setView();
        $this->$action();
    }
    
    protected function setView(){
        $this->view = new view;
    }

    
    protected function getControllerName(){
        return basename(str_replace('\\', '/', get_class($this)));
    }
    

    protected function html(){
        $this->view->assign("script",script::getScript());
        $this->view->add($this->layout);
        $this->view->render();
    }

    protected function error($error){
        utils::log($error);    
        $this->errorMsg[] = $error;
        $this->view->assign("errors",$this->errorMsg);
        $this->view->error = true;
    }
    public function __toString(){
        return $this->getControllerName();
    }

}