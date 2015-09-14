<?php

/**
 * Description of component
 *
 * @author durso
 */
namespace library\dom\structures;
use library\dom\object;
use library\dom\elements\components\elementFactory;
use library\dom\elements\elementCollection;


abstract class components{
    protected $root;

    public function __construct($tag){
        $this->root = elementFactory::createByTag($tag);
    }
    public function addComponent(object $component){
        $this->root->addComponent($component);
    }

    public function save(){
        return $this->root;
    }

    public function __call($name, $arguments) {
        return call_user_func_array(array($this->root, $name), $arguments);   
    }
}

