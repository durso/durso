<?php

namespace app\controller\base; 
use app\view\view;
//use app\model\db;
use app\model\file;
use library\utils;
use library\layout\elements\script;
use library\layout\elements\element;



class controller{
    protected $query;
    protected $action;
    protected $render = array();
    protected $errorMsg = array();
    protected $view;
    protected $controller;
    
    protected function __construct($action,$query) {
        $this->controller = $this->getControllerName();
        $this->query = $query;
        $this->action = $action;
        $this->setView();
        script::setAction($action);
        script::setController($this->controller);
        $this->$action();
    }
    
    protected function setView(){
        $this->view = new view;
    }

    
    protected function getControllerName(){
        return basename(str_replace('\\', '/', get_class($this)));
    }
    
    
    protected function addTemplate($template, $hasPath = false)
    {
        if(!$hasPath){
            $file = TEMPLATE_PATH.DS.$template.".php";
        } else {
            $file = $template;
        }
        if (file::isReadable($file)) {
            $this->render[] = $file;
        } else {
            $error = "Could not render the template: ".$file;
            utils::log($error);
            $this->errorMsg[] = "Could not render the template";
            $this->view->assign("errors",$this->errorMsg);
            $this->view->apologize = true;
        }
    }
    
    protected function addView() {
        $file = VIEW_PATH . DS . $this->controller . DS . $this->action . '.php';  
        $this->addTemplate($file,true);
    }
    protected function html(element $element){
        $this->view->assign("script",script::getScript());
        $this->view->add($element);
        $this->addView();
        $this->view->files($this->render);
        $this->view->render();
    }

    
    public function __toString(){
        return $this->getControllerName();
    }

}